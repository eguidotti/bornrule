import math

try:
    import torch

except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "PyTorch required but not installed. "
        "Please install PyTorch at https://pytorch.org")


class Born(torch.nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Born, self).__init__()

        rho = 1. / math.sqrt(in_features)
        theta = 2. * math.pi * torch.rand(in_features, out_features)

        real = rho * torch.cos(theta)
        imag = rho * torch.sin(theta)

        if dtype is None:
            dtype = torch.get_default_dtype()

        weight = torch.complex(real, imag) if self.is_complex(dtype) else torch.stack((real, imag))
        weight = weight.to(device=device, dtype=dtype)
        self.weight = torch.nn.Parameter(weight)
        
    def forward(self, x):

        if self.is_complex(self.weight.dtype):
            proba = torch.pow(torch.mm(x, self.weight).abs(), 2)

        else:
            real = torch.mm(x, self.weight[0])
            imag = torch.mm(x, self.weight[1])
            proba = torch.pow(real, 2) + torch.pow(imag, 2)

        return torch.nn.functional.normalize(proba, p=1)

    @staticmethod
    def is_complex(dtype):
        return dtype.is_complex if hasattr(dtype, 'is_complex') else False
