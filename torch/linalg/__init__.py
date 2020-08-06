import sys

import torch
from torch._C import _add_docstr, _linalg  # type: ignore

Tensor = torch.Tensor

# Note: This not only adds the doc strings for the spectral ops, but
# connects the torch.fft Python namespace to the torch._C._fft builtins.

outer = _add_docstr(_linalg.linalg_outer, r"""
linalg.outer(input, vec2, *, out=None) -> Tensor

Alias of :func:`torch.ger`.
""")
