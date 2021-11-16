# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import tensorflow as tf
import torch
import torch.nn as nn

class TopDownAttentionTorch(nn.Module):
    def __init__(self, units, **kwarg):
        super().__init__()
        self.units   = units
        self.built = False

    def build(self, input):
        self.w1 = nn.Sequential(
            nn.Linear(input.size(-1), self.units),
            nn.Tanh()
        )
        
        self.w2 = nn.Sequential(
            nn.Linear(self.units, self.units),
            nn.Sigmoid()
        )

        self.wt = nn.Linear(self.units, 1, bias=False)

        self.softmax = nn.Softmax(-1)

    def forward(self, inputs):

        language = inputs[0]
        features = inputs[1]
        num_regions = features.size(1)
        language = (language.unsqueeze(1)).repeat(1,num_regions,1)           # bxkxm
        att_in   = torch.cat((language, features), axis=2) # bxkx(m+n)

        if not self.built:
            self.build(att_in)
            
        y_1 = self.w1(att_in)
        y_2 = self.w2(y_1)
        y   = torch.multiply(y_1, y_2)
        a   = self.wt(y)
        a   = a.squeeze()

        return self.softmax(a)
        