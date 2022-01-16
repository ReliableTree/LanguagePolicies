from copy import deepcopy
import torch
import torch.nn as nn

class feature_net(nn.Module):
    def __init__(self, arc) -> None:
        super().__init__()
        module_list = []
        self.arc = arc
        for i in range(len(self.arc) - 2):
            module_list.append(nn.Linear(self.arc[i], self.arc[i+1]))
            module_list.append(nn.ReLU())
        module_list.append(nn.Linear(self.arc[-2], self.arc[-1]))
        #module_list.append(nn.Sigmoid())
        self.network = nn.Sequential(*module_list)

    def forward(self, inpt):
        return self.network(inpt)

class CNet2(nn.Module):
    def __init__(self, arc, device = 'cpu') -> None:
        super().__init__()
        self.arc = arc
        self.device = device
        self.down_net = feature_net(arc[:-1]).to(device)
        self.up_net = feature_net(arc[-2:]).to(device)
        print(f'new arc: {self.arc}')

    def make_feature(self):
        self.arc[-2] += 1
        self.down_net = feature_net(self.arc[:-1]).to(self.device)
        self.up_net = feature_net(self.arc[-2:]).to(self.device)
        print(f'new arc: {self.arc}')
        
    def forward(self, inpt):
        inter_res = self.down_net(inpt)
        '''with torch.no_grad():
            between = inter_res
            between[between>0.5] = 1
            between[between<=0.5] = 0
            inter_res = inter_res - inter_res + between
        print(f'inter_res: {inter_res}')'''
        result = self.up_net(inter_res)
        return result
