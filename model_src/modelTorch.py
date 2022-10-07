from abc import abstractclassmethod
import typing
import torch
import torch.nn as nn
from LanguagePolicies.utils.Transformer import CriticTransformer, TransformerModel
from MetaWorld.utilsMW.model_setup_obj import ModelSetup
from MetaWorld.searchTest.utils import calcMSE


class WholeSequenceModel(nn.Module):
    def __init__(self, model_setup: ModelSetup) -> None:
        super().__init__()
        self.model_setup = model_setup
        self.model: TransformerModel = None
        self.optimizer: torch.optim.Optimizer = None

    def init_model(self):
        self.model = None
        self.optimizer = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if (self.model is None):
            self.model_setup.ntoken = inputs.size(-1)
            self.model = self.model_setup.model_class(
                model_setup=self.model_setup).to(inputs.device)
        result = self.model.forward(inputs)
        if self.optimizer is None:
            self.optimizer = self.model_setup.optimizer_class(
                self.model.parameters(), self.model_setup.lr, **self.model_setup.optimizer_kwargs)
        return result

    @abstractclassmethod
    def optimizer_step(self, data: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> typing.Dict:
        raise NotImplementedError


class WholeSequenceActor(WholeSequenceModel):
    def __init__(self, model_setup: ModelSetup):
        super().__init__(model_setup)

    def loss_fct(self, result, label) -> torch.Tensor:
        trj_loss = calcMSE(result, label)
        return trj_loss

    def optimizer_step(self, data: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor], prefix='') -> typing.Dict:
        inputs, label, success = data
        assert torch.all(success), 'wrong trajectories input to Actor Model'
        results = self.forward(inputs=inputs)
        loss = self.loss_fct(result=results, label=label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        debug_dict = {
            'Trajectory Loss '+ prefix: loss.detach()
        }

        return debug_dict



class WholeSequenceCritic(WholeSequenceModel):
    def __init__(self, model_setup: ModelSetup):
        super().__init__(model_setup)

    def init_model(self):
        super().init_model()

    def loss_fct(self, inpt, success):
        success = success.reshape(-1).type(torch.float)
        inpt = inpt.reshape(-1).type(torch.float)
        loss = (inpt - success)**2
        success = success.type(torch.bool)
        label_sum = success.sum()
        if label_sum > 0:
            loss_positive = loss[success].mean()
        else:
            loss_positive = torch.zeros(1, device=inpt.device).mean()
        if label_sum < len(success):
            loss_negative = loss[~success].mean()
        else:
            loss_negative = torch.zeros(1, device=inpt.device).mean()
        loss = loss.mean()
        return loss, loss_positive, loss_negative

    def optimizer_step(self, data: typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> typing.Dict:
        debug_dict = {}
        inputs, label, success = data
        scores = self.forward(inputs=inputs)
        loss, loss_positive, loss_negative = self.loss_fct(
            inpt=scores, success=success)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        debug_dict['critic loss'] = loss.detach()
        debug_dict['critic loss positive'] = loss_positive.detach()
        debug_dict['critic loss negative'] = loss_negative.detach()

        return debug_dict
