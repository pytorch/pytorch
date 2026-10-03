import torch
from torch.overrides import is_tensor_like


maybe_tensor: object = torch.tensor([1.0])
if is_tensor_like(maybe_tensor):
    reveal_type(maybe_tensor)  # E: {Tensor}

# The guard intersects rather than replaces, so a declared subclass survives it.
param = torch.nn.Parameter(torch.tensor([1.0]))
if is_tensor_like(param):
    reveal_type(param)  # E: torch.nn.parameter.Parameter
