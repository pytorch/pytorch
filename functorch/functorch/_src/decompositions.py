import torch
from torch import Tensor
import torch._decomp

aten = torch.ops.aten

decomposition_table = torch._decomp.decomposition_table
register_decomposition = torch._decomp.register_decomposition
get_decompositions = torch._decomp.get_decompositions


@register_decomposition(aten.trace.default)
def trace(self: Tensor) -> Tensor:
    return torch.sum(torch.diag(self))

# Decompositions have been ported to torch._decomp inside of PyTorch core. The only decompositions here are temporary or hacks. Please provide your contributions to PyTorch core!
