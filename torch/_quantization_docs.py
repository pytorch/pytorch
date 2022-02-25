"""Adds docstrings to functions in the torch.ops.quantized namespace"""

import torch._C
from torch._C import _add_docstr as add_docstr

# TODO(future PR): extend this docblock and make it render in the html docs
add_docstr(torch.ops.quantized.add,
           """
torch.ops.quantized.add(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc

This is the quantized version of `torch.add`.
""")

# TODO(future PR): document the other developer facing functions from
# torch.ops.quantized
