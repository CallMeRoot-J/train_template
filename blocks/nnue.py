import torch
import torch.nn as nn


class NNUE(nn.Module):
    def __init__(self, in_features: int):
        super(NNUE, self).__init__()
        self.input = nn.Linear(in_features=in_features, out_features=in_features)
        self.l1 = nn.Linear(in_features, in_features)
        self.l2 = nn.Linear(in_features, in_features)
        self.output = nn.Linear(in_features, 3)

    def forward(self, x):
        x = torch.clamp(x, 0.0, 127 / 128)
        state = torch.clamp(self.input(x), 0.0, 127 / 128)
        state = torch.clamp(self.l1(state), 0.0, 127 / 128)
        state = torch.clamp(self.l2(state), 0.0, 127 / 128)
        v = self.output(state)
        return v
