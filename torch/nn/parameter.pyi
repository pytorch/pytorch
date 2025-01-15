from typing_extensions import TypeIs

from torch import device, dtype, Tensor

class Parameter(Tensor):
    def __init__(self, data: Tensor = ..., requires_grad: bool = ...) -> None: ...

def is_lazy(
    param: Tensor,
) -> TypeIs[UninitializedParameter | UninitializedBuffer]: ...

class UninitializedParameter(Tensor):
    def __init__(self, data: Tensor = ..., requires_grad: bool = ...) -> None: ...
    def materialize(
        self,
        shape: tuple[int, ...],
        device: device | None = None,
        dtype: dtype | None = None,
    ) -> None: ...

class Buffer(Tensor):
    persistent: bool
    def __init__(
        self,
        data: Tensor = ...,
        requires_grad: bool = ...,
        persistent: bool = ...,
    ): ...

class UninitializedBuffer(Tensor):
    persistent: bool
    def __init__(
        self,
        data: Tensor = ...,
        requires_grad: bool = ...,
        persistent: bool = ...,
    ): ...
    def materialize(
        self,
        shape: tuple[int, ...],
        device: device | None = None,
        dtype: dtype | None = None,
    ) -> None: ...
