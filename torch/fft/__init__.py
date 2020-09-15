import sys

import torch
from torch._C import _add_docstr, _fft  # type: ignore

Tensor = torch.Tensor

# Note: This not only adds the doc strings for the spectral ops, but
# connects the torch.fft Python namespace to the torch._C._fft builtins.

fft = _add_docstr(_fft.fft_fft, r"""
fft(input) -> Tensor

Computes the one dimensional discrete Fourier transform of :attr:`input`.

Args:
  input (Tensor): the input tensor

Example::

    >>> t = torch.randn(4, dtype=torch.complex128)
    >>> t
    tensor([-1.1364-0.5694j, -0.6637+0.9987j, -1.0102-0.4383j,  0.3017+0.9371j], dtype=torch.complex128)
    >>> torch.fft.fft(t)
    tensor([-2.5086+0.9281j, -0.0646+0.8343j, -1.7846-2.9435j, -0.1878-1.0965j], dtype=torch.complex128)
""")
