import torch
import torch.nn as nn
class dmp_dt_torch(nn.Module):
    def __init__(self, ptgloabl_units, units) -> None:
        super().__init__()
        self.ptgloabl_units = ptgloabl_units
        self.units = units


    def build_dmp_dt_model(self, input):
            pre_layers = [
                nn.Linear(input.size(-1), self.ptgloabl_units),
                nn.ReLU()
            ]
            
            post_layers = [
                nn.Linear(self.ptgloabl_units, self.units * 2),
                nn.ReLU(),
                nn.Linear(self.units * 2, 1),
                nn.Hardsigmoid()
            ]

            self.dmp_dt_model_pre = nn.ModuleList(pre_layers).to(input.device)
            self.dmp_dt_model_post = nn.ModuleList(post_layers).to(input.device)

            def build_model(inpt, use_dropout):
                result = inpt
                for l in self.dmp_dt_model_pre:
                    result = l(result)

                if use_dropout:
                    print('used dropout')
                    result = nn.Dropout(p=0.25)(result)

                for l in self.dmp_dt_model_post:
                    result = l(result)
                return result + 0.1

            self.dmp_model = build_model