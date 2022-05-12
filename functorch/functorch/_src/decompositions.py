import torch
from torch import Tensor
import torch._decomp
from typing import Tuple

aten = torch.ops.aten

decomposition_table = torch._decomp.decomposition_table
register_decomposition = torch._decomp.register_decomposition
get_decompositions = torch._decomp.get_decompositions

# Decompositions have been ported to torch._decomp inside of PyTorch core. The only decompositions here are temporary or hacks. Please submit your contributions to PyTorch core!


@register_decomposition(aten.trace.default)
def trace(self: Tensor) -> Tensor:
    return torch.sum(torch.diag(self))


@register_decomposition(aten.log_sigmoid_forward)
def log_sigmoid_forward(self: Tensor) -> Tuple[Tensor, Tensor]:
    min = torch.minimum(self.new_zeros(()), self)
    z = torch.exp(-torch.abs(self))
    if self.is_cuda:
        buffer = self.new_zeros((0,))
    else:
        buffer = z
    return min - torch.log1p(z), buffer
