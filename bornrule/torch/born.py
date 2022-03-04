try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "PyTorch required but not installed. "
        "Please install PyTorch at https://pytorch.org")

import math


class Born(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        rho = 1. / math.sqrt(in_features)
        theta = 2. * math.pi * torch.rand(in_features, out_features, device=device, dtype=dtype)
        self.weight = torch.nn.Parameter(torch.complex(real=rho*torch.cos(theta), imag=rho*torch.sin(theta)))
        
    def forward(self, x):
        return torch.nn.functional.normalize(torch.pow(torch.mm(x, self.weight).abs(), 2), p=1)
