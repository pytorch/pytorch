# mypy: allow-untyped-defs
from typing_extensions import TypeGuard

from torch import device, dtype, Tensor

class Parameter(Tensor):
    def __init__(self, data: Tensor = ..., requires_grad: bool = ...) -> None: ...

def is_lazy(
    param: Tensor,
) -> TypeGuard[UninitializedParameter | UninitializedBuffer]: ...

class UninitializedParameter(Tensor):
    def __init__(self, data: Tensor = ..., requires_grad: bool = ...) -> None: ...
    def materialize(
        self,
        shape: tuple[int, ...],
        device: device | None = None,
        dtype: dtype | None = None,
    ) -> None: ...

class UninitializedBuffer(Tensor):
    def __init__(self, data: Tensor = ..., requires_grad: bool = ...) -> None: ...
    def materialize(
        self,
        shape: tuple[int, ...],
        device: device | None = None,
        dtype: dtype | None = None,
    ) -> None: ...
