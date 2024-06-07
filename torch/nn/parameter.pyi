import builtins
from typing import Optional, Tuple

import torch
from torch import Tensor

class Parameter(Tensor):
    def __init__(
        self,
        data: Tensor = ...,
        requires_grad: builtins.bool = ...,
    ): ...

def is_lazy(param: Tensor): ...

class UninitializedParameter(Tensor):
    def __init__(
        self,
        data: Tensor = ...,
        requires_grad: builtins.bool = ...,
    ): ...
    def materialize(
        self,
        shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ): ...

class UninitializedBuffer(Tensor):
    def __init__(
        self,
        data: Tensor = ...,
        requires_grad: builtins.bool = ...,
    ): ...
    def materialize(
        self,
        shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ): ...
