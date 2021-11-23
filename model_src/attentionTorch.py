# @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import tensorflow as tf
import torch
import torch.nn as nn

class TopDownAttentionTorch(nn.Module):
    def __init__(self, units, **kwarg):
        super().__init__()
        self.units   = units
        self.attn_model = None

    def build_attn_model(self, input):
        self.w1 = nn.Sequential(
            nn.Linear(input.size(-1), self.units),
            nn.Tanh()
        ).to(input.device)

        self.w2 = nn.Sequential(
            nn.Linear(input.size(-1), self.units),
            nn.Sigmoid()
        ).to(input.device)

        self.wt = nn.Linear(self.units, 1, bias=False).to(input.device)

        self.softmax = nn.Softmax(-1)

        def model(att_in):
            y_1 = self.w1(att_in)
            y_2 = self.w2(att_in)
            y   = y_1 * y_2
            a   = self.wt(y)
            '''print('dimensions in attention Torch:')
            print(f'att_in {att_in.shape}')
            print(f'y_1 {y_1.shape}')
            print(f'y_2 {y_2.shape}')
            print(f'a {a.shape}')'''
            a   = a.squeeze()
            #print(f'a nach squeeze {a.shape}')
            return self.softmax(a)

        self.attn_model = lambda inpt: model(inpt)




    def forward(self, inputs):

        language = inputs[0]
        features = inputs[1]
        '''print(f'language {language.shape}')
        print(f'features {features.shape}')'''

        num_regions = features.size(1)
        language = (language.unsqueeze(1)).repeat(1,num_regions,1)           # bxkxm
        #print(f'language {language.shape}')
        att_in   = torch.cat((language, features), axis=2) # bxkx(m+n)

        if self.attn_model is None:
            self.build_attn_model(att_in)

        return self.attn_model(att_in)
            
        