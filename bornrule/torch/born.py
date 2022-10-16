import math

try:
    import torch

except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "PyTorch required but not installed. "
        "Please install PyTorch at https://pytorch.org")


class Born(torch.nn.Module):
    r"""Pytorch implementation of Born's Layer

    This class is compatible with [pytorch](https://pytorch.org).
    It supports real and complex-valued inputs. Outputs probabilities in the range $`[0, 1]`$.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    device : torch.device
        The [device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
        on which `weight` is allocated.
    dtype : torch.dtype
        The [data type](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)
        of `weight`.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable complex-valued weights of the module. The values are initialized from:

        ```math
        \frac{e^{i\theta}}{\sqrt{S}} \quad \text{with} \quad \theta \sim \mathcal{U}(0,2\pi).
        ```

        where $`S`$ is equal to `in_features`, and $`i`$ is the imaginary unit.
        The shape is (`in_features`, `out_features`) when `dtype` is a complex type.
        Otherwise, the shape is (`2`, `in_features`, `out_features`) where the first
        dimension stores the real and imaginary parts, respectively.

    """

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
        r"""Applies the following transformation to the incoming data:

        ```math
        y = \dfrac{\operatorname{Mod}(xW)^2}{|| \operatorname{Mod}(xW)^2 ||_1}
        ```

        where $`\operatorname{Mod}`$ is the modulus of complex numbers,
        and $`||\cdot||_1`$ is the L1-norm of a vector.

        Parameters
        ----------
        x : torch.Tensor
            Input samples of shape (`n_samples`, `in_features`).

        Returns
        -------
        y : torch.Tensor
            Output probabilities of shape (`n_samples`, `out_features`).

        """
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
