import sys

import torch
from torch._C import _add_docstr, _linalg  # type: ignore

Tensor = torch.Tensor

# Note: This not only adds doc strings for the linear algebra ops, but
# also connects the torch.linalg Python namespace to the torch._C._linalg builtins.

outer = _add_docstr(_linalg.linalg_outer, r"""
linalg.outer(input, vec2, *, out=None) -> Tensor

Alias of :func:`torch.ger`.
""")
