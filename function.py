import torch


class RadialBasis:
    def __init__(self):
        self.m = 1.
        self.sigma = 1.

    def __call__(self, x, *args, **kwargs):
        return torch.exp(torch.pow(torch.abs(self.m - x), 2) / 2 * self.sigma)
