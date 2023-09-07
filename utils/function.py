import torch


class RadialBasis:
    def __call__(self, x, *args, **kwargs):
        return torch.exp(-torch.pow(torch.abs(torch.mean(x) - x), 2) / 2 * torch.std(x))
