import torch
import torch.nn as nn
from utils.Transformer import TransformerModel


class TransformerAttention(nn.Module):
    def __init__(self, device, d_model: int = 50, nhead: int = 1, d_hid: int = 50,nlayers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.dropout = dropout
        self.seq_embedding = None
        self.TA = None#TransformerModel(ntoken=ntoken, d_output=dropout, d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers, dropout=dropout)

    def forward(self, inpt):
        language, features = inpt
        language = language.unsqueeze(1).repeat([1,features.size(1),1])
        #print(f'language size: {language.shape}')
        #print(f'features size: {features.shape}')
        #features = 16x6x5
        #language = 16x32
        lf = torch.cat((features, language), dim = 2) #16x6x37
        lf = lf.transpose(0,1) #6X16X37
        if self.TA is None:
            self.TA = TransformerModel(ntoken=lf.size(-1), d_output=lf.size(0), d_model=self.d_model, nhead=self.nhead, d_hid=self.d_hid, nlayers=self.nlayers, dropout=self.dropout).to(self.device)
        attn = self.TA(lf)                     #6x16x6
        #print(f'attn size: {attn.shape}')

        attn = attn.transpose(0,2)             #16x6x6, batch,embedding,sequence
        if self.seq_embedding is None:
            self.seq_embedding = nn.Linear(attn.size(-1), 1).to(self.device)
        attn = self.seq_embedding(attn)
        attn = attn.squeeze().transpose(0,1)                  #16x6
        #print(f'attn size end tras: {attn.shape}')
        attn = nn.Softmax(dim = -1)(attn)
        #print(f'attn size end tras: {attn.shape}')
        return attn