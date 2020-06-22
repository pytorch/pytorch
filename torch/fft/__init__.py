import sys

import torch
from torch._C import _add_docstr

# Note [Callable Modules]
#   Modules in Python3 cannot be made callable. However, Guido suggests
#   https://mail.python.org/pipermail/python-ideas/2012-May/014969.html
#   where a callable class acts replaces an imported module.
#   Our interest in torch.fft being callable is to let users call
#   torch.fft() the function as they have in previous versions of PyTorch.
class fft_class:
    # Mimics torch.fft()
    __doc__ = torch.fft.__doc__

    def __call__(self, *args, **kwargs):
        return torch.fft(*args, **kwargs)

    # Adds functions in the torch.fft namespace
    fft = torch._C.fft.fft

# See Note [Callable Modules]
sys.modules[__name__] = fft_class()

# docstring registrations for functions in the torch.fft namespace

fft = _add_docstr(fft_class.fft, r"""
fft(input) -> Tensor

Applies the discrete Fourier transform to the complex input, returning
a complex output.
""")
