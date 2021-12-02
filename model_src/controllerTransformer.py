import torch
import torch.nn as nn
from utils.Transformer import TransformerModel

class ControllerTransformer(nn.Module):
    def __init__(self, ntoken: int, d_output: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.TF = TransformerModel(ntoken, d_output, d_model, nhead, d_hid,
                 nlayers, dropout)
        
    def forward(self, seq_inputs, src_mask ,training=False):
        return self.TF(seq_inputs, src_mask= src_mask)
        #TODO validation time with self imputs

        actions_seq = None
        phase_seq = None
        weights_seq = None
        initilaized = False
        for event in range(int(seq_inputs.size(1))):
            (action, phase, weights), gru_output = self.Cell.forward(inputs=seq_inputs[:,event,:], states=states, constants=constants, training=training)
            states = (action, gru_output)
            if not initilaized:
                actions_seq = torch.zeros_like(seq_inputs)
                phase_seq = torch.zeros([seq_inputs.size(0), seq_inputs.size(1), 1], device = seq_inputs.device, dtype=seq_inputs.dtype)
                weights_seq = torch.zeros([seq_inputs.size(0), seq_inputs.size(1), weights.size(1), weights.size(2)], device = seq_inputs.device, dtype=seq_inputs.dtype)
                initilaized = True
                
            actions_seq[:,event,:] = action
            phase_seq[:,event,:] = phase
            weights_seq[:,event,:] = weights
        return actions_seq, phase_seq, weights_seq, gru_output