import torch
import torch.nn as nn
from utils.Transformer import TransformerModel

class TorchTranspose(nn.Module):
    def __init__(self, transpose) -> None:
        super().__init__()
        self.transpose = transpose

    def forward(self, inpt):
        result = inpt
        for transp in self.transpose:
            result = result.transpose(*transp)
        return result
        
class TransformerUpConv(nn.Module):
    def __init__(self, num_upconvs, stride, ntoken: int, d_output: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dilation = 2, dropout: float = 0.5, seq_len = 350, use_layernorm=True, upconv = True) -> None:
        super().__init__()
        self.seq_len = seq_len
        module_list = nn.ModuleList()
        for i in range(num_upconvs):
            if i == 0:
                module_list += [
                    TorchTranspose([(0,1), (1,2)])
                ]
                if upconv:
                    module_list += [
                    nn.ConvTranspose1d(in_channels=ntoken, out_channels=ntoken, kernel_size=stride, stride=stride, dilation=dilation)
                    ]
                else:
                    module_list += [
                        nn.Conv1d(in_channels=ntoken, out_channels=ntoken, kernel_size=stride, stride=stride)
                    ]
                module_list += [
                    TorchTranspose([(0,1), (0,2)]),
                    TransformerModel(ntoken=ntoken, d_output=d_model, d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers),
                ]
                if use_layernorm:
                        module_list += [nn.LazyBatchNorm1d()]
            elif i == num_upconvs-1:
                module_list += [
                    TorchTranspose([(0,1), (1,2)]),
                ]

                if upconv:
                    module_list += [
                        nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, kernel_size=stride, stride=stride)
                    ]
                else:
                    module_list += [
                        nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=stride, stride=stride)
                    ]
                module_list += [
                    TorchTranspose([(0,1), (0,2)]),
                    TransformerModel(ntoken=d_model, d_output=d_output, d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers)
                ]
                if use_layernorm:
                    module_list += [nn.LazyBatchNorm1d()]
            else:
                module_list += [
                    TorchTranspose([(0,1), (1,2)]),
                ]

                if upconv:
                    module_list += [
                        nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, kernel_size=stride, stride=stride)
                    ]
                else:
                    module_list += [
                        nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=stride, stride=stride)
                    ]
                module_list += [
                    TorchTranspose([(0,1), (0,2)]),
                    TransformerModel(ntoken=d_model, d_output=d_model, d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers)  
                ]
        self.module_list = module_list

    def forward(self, inpt):
        result = inpt
        for module in self.module_list:
            result = module(result)
        if (result.size(0) - self.seq_len) %2 ==0:
            cropping = int((result.size(0) - self.seq_len)/2)
            return result[cropping : -cropping]
        else:
            cropping = int((result.size(0) - self.seq_len - 1)/2)
            return result[cropping: -(cropping + 1)]