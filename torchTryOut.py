import torch
import torch.nn as nn

'''class BasisModelTorch(nn.Module):
    def __init__(self, nfunctions, scale, **kwargs):
        super().__init__()
        self._degree = nfunctions
        #self._degree = nfunctions
        self._scale = scale
        self.register_buffer('_centers' ,torch.linspace(0.0, 1.01, self._degree))

    def apply_basis_model(self, inputs):
        weights     = inputs[0]#.transpose(-2,-1)
        positions   = inputs[1]
        positions = positions.unsqueeze(-1)
        basis_funcs = self.compute_basis_values(positions)
        result      = torch.matmul(basis_funcs, weights)

        return result

    def compute_basis_values(self, X):
        return torch.exp(-(X-self._centers)**2/(2.0*self._scale))

if __name__ == '__main__':
    num_basis = 100
    num_positions = 10
    epochs = 1000
    points = torch.rand(num_positions)
    basis_model = BasisModelTorch(num_basis, 1)
    weights = torch.rand(num_basis, requires_grad=True)
    positions = torch.linspace(0,1,num_positions)
    optim = torch.optim.SGD([weights], lr=1e-7)
    best_loss = float('inf')
    for i in range(epochs):
        computes_positions = basis_model.apply_basis_model([weights, positions])
        loss = ((points - computes_positions)**2).sum()
        loss.backward()
        optim.step()
        if loss < best_loss:
            best_weights = torch.clone(weights.detach())
            best_loss = loss
            #print(loss)
            #print(positions)
            print(basis_model._centers)
    computes_positions = basis_model.apply_basis_model([best_weights, positions])
    #print(points)
    #print(((points - computes_positions)**2).sum())'''
