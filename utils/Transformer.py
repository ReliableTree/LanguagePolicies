import math
from statistics import mode
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ModelSetup:
    def __init__(self) -> None:
        self.use_layernorm = False
        self.upconv = True
        self.num_upconvs = 5
        self.stride = 3
        self.d_output = 4
        self.nhead = 4
        self.d_hid = 512
        self.d_model = 512
        self.nlayers = 4
        self.seq_len = 100
        self.dilation = 2
        self.d_result = None
        self.ntoken = -1
        self.dropout = 0.2
        self.lr = None
        self.device = 'cuda'
        self.optimizer_class = torch.optim.AdamW
        self.optimizer_kwargs = {}
        self.model_class: TransformerModel = TransformerModel


class TransformerModel(nn.Module):

    def __init__(self, model_setup: ModelSetup = None):
        super().__init__()
        self.model_type = 'Transformer'
        ntoken = model_setup.ntoken
        d_output = model_setup.d_output
        d_model = model_setup.d_model
        nhead = model_setup.nhead
        d_hid = model_setup.d_hid
        nlayers = model_setup.nlayers
        dropout = model_setup.dropout

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, d_output)
        self.mask = generate_square_subsequent_mask(
            sz=model_setup.seq_len).to('cuda')

        # self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask=None) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(
            src, self.mask[:src.size(0), :src.size(0)])
        output = self.decoder(output)
        return output


class TailorTransformer(TransformerModel):
    def __init__(self, model_setup: ModelSetup):
        self.super_init = False
        self.model_setup = model_setup

    def forward(self, src: Tensor) -> Tensor:
        if not self.super_init:
            self.model_setup.ntoken = src.size(-1)
            super().__init__(model_setup=self.model_setup)
            self.result_encoder = nn.Linear(
                self.model_setup.d_output * self.model_setup.seq_len, self.model_setup.d_result)
            #self.sm = torch.nn.Softmax(dim=-1)
            self.to(src.device)
            self.super_init = True

        # src: batch, seq, dim
        #src = seq, batch, dim
        #print(f'src shape: {src.shape}')
        pre_result = super().forward(src)

        #preresult = batch, seq, dim
        pre_result = pre_result.reshape(pre_result.size(0), -1)
        #print(f'preresult shape {pre_result.shape}')

        result = self.result_encoder(pre_result)
        #result = self.sm(result)
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
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
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


def get_batch(source: Tensor, i: int, bptt: int) -> Tuple[Tensor, Tensor]:
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
