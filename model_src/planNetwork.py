import torch
import torch.nn as nn

class Plan_NN(nn.Module):
    def __init__(self, inpt_size, outpt_size) -> None:
        super().__init__()
        lin1 = nn.Linear(inpt_size, outpt_size)
        lin2 = nn.Linear(outpt_size, outpt_size)
        self.model = nn.Sequential(
            lin1,
            nn.ReLU(),
            lin2
        )

    def forward(self, inpt):
        return self.model(inpt)
