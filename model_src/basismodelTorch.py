# close to @author Simon Stepputtis <sstepput@asu.edu>, Interactive Robotics Lab, Arizona State University

import torch
import torch.nn as nn

class BasisModelTorch(nn.Module):
    def __init__(self, nfunctions, scale, **kwargs):
        super().__init__()
        self._degree = nfunctions
        #self._degree = nfunctions
        self._scale = scale
        self.register_buffer('_centers' ,torch.linspace(0.0, 1.01, self._degree))

    def apply_basis_model(self, inputs):
        weights     = inputs[0].transpose(-2,-1)
        positions   = inputs[1]
        positions = positions.unsqueeze(-1)
        basis_funcs = self.compute_basis_values(positions)
        result      = torch.matmul(basis_funcs, weights)

        return result

    def compute_basis_values(self, X):
        return torch.exp(-(X-self._centers)**2/(2.0*self._scale))