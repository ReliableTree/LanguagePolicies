# close to @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import torch
import torch.nn as nn

class BasisModelTorch(nn.Module):
    def __init__(self, nfunctions, scale, **kwargs):
        super().__init__()
        self._degree = nfunctions
        #self._degree = nfunctions
        self.scale = scale
        self.centers = torch.linspace(0.0, 1.01, self._degree, device = 'cuda')
        

    def apply_basis_model(self, inputs):
        weights     = inputs[0].transpose(-2,-1)
        positions   = inputs[1]
        positions = positions.unsqueeze(-1)
        basis_funcs = self.compute_basis_values(positions)
        result      = torch.matmul(basis_funcs, weights)
        '''print(f'weights shape: {weights.shape}')
        print(f'basis_funcs shape: {basis_funcs.shape}')
        print(f'result shape: {result.shape}')'''
        return result

    def compute_basis_values(self, x):
        centers = self.centers.unsqueeze(0).repeat(x.size(1), 1)
        funcs = torch.exp(-((x-centers)**2)/(2*self.scale))
        '''print(f'centers shape: {centers.shape}')
        print(f'x shape: {x.shape}')
        print(f'funcs shape: {funcs.shape}')'''
        return funcs
