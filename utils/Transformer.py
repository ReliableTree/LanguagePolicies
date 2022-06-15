import math
from turtle import forward
from typing import Tuple
from importlib_metadata import version

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):

    def __init__(self, ntoken = 0, d_output = 0, d_model = 0, nhead = 0, d_hid = 0,
                 nlayers = 0, dropout = 0.2, model_setup = None):
        super().__init__()
        self.model_type = 'Transformer'
        if model_setup is not None:
            ntoken = model_setup['ntoken']
            #d_output = model_setup['d_output']
            d_model = model_setup['d_model']
            nhead = model_setup['nhead']
            d_hid = model_setup['d_hid']
            nlayers = model_setup['nlayers']
            d_output = 4
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, d_output)

        #self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask = None) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        #print(src[:,0])
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        #print(output[:,0])
        #print('______________________')
        #output = self.decoder(output)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, model_setup, verbose = False) -> None:
        super().__init__()
        self.d_output = model_setup['d_output']
        self.super_init = False
        self.output_seq = model_setup['output_seq']
        self.verbose = verbose

    def forward(self, inpt):
        #inpt N,S,D
        if not self.super_init:
            if self.output_seq:
                self.decoder = nn.Linear(inpt.size(-1), self.d_output)
            else:
                self.decoder = nn.Linear(inpt.size(-1)*inpt.size(-2), self.d_output)
            self.decoder.to(inpt.device)
            self.super_init = True

        if not self.output_seq:
            inpt_dec = inpt.reshape(inpt.size(0), -1)
            output = self.decoder(inpt_dec)
            output = output.reshape(-1)
        else:
            inpt_dec = inpt
            output = self.decoder(inpt_dec)
        
        return output


class TailorTransformer(TransformerModel):
    def __init__(self, ntoken=0, d_output=0, d_model=0, nhead=0, d_hid=0, nlayers=0, dropout=0.2, model_setup=None):
        self.super_init = False
        self.model_setup = model_setup
        

    def forward(self, src: Tensor) -> Tensor:
        if not self.super_init:
            self.model_setup['ntoken'] = src.size(-1)
            self.model_setup['seq_len'] = src.size(1)
            super().__init__(model_setup = self.model_setup)
            self.result_encoder = nn.Linear(self.model_setup['d_output'] * self.model_setup['seq_len'], self.model_setup['d_result'])
            #self.sm = torch.nn.Softmax(dim=-1)
            self.sig = torch.nn.Sigmoid()
            self.ReLU = torch.nn.ReLU()
            self.to(src.device)
            self.super_init = True

        #src: batch, seq, dim
        src = src.transpose(0,1)

        #src = seq, batch, dim
        #print(f'src shape: {src.shape}')
        pre_result = super().forward(src)

        pre_result = pre_result.transpose(0,1)

        #preresult = batch, seq, dim
        pre_result = pre_result.reshape(pre_result.size(0), -1)
        #print(f'preresult shape {pre_result.shape}')

        result = self.ReLU(self.result_encoder(pre_result)) 
        #result = (self.sig(result) + 1) /2
        #print(f'result shape {result.shape}')

        return result

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def batchify(data: Tensor, bsz: int, device: str) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

def get_batch(source: Tensor, i: int, bptt : int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

