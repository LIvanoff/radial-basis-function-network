import torch.nn as nn
from utils.function import RadialBasis


class RBF(nn.Module):
    def __init__(self, input_n, output_n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(input_n, output_n)
        self.act = RadialBasis()
        self.fc2 = nn.Linear(output_n, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        return out

