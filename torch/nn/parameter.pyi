from typing_extensions import TypeGuard

from torch import dtype, Tensor
from torch._C import _TensorMeta
from torch.types import Device

class _ParameterMeta(_TensorMeta):
    def __instancecheck__(self, instance: object) -> bool: ...

class Parameter(Tensor, metaclass=_ParameterMeta):
    def __init__(
        self,
        *,
        data: Tensor | None = None,
        requires_grad: bool = True,
    ) -> None: ...

def is_lazy(
    param: Tensor,
) -> TypeGuard[UninitializedParameter | UninitializedBuffer]: ...

class UninitializedParameter(Parameter):
    def __init__(
        self,
        requires_grad: bool = True,
        device: Device = None,
        dtype: dtype | None = None,
    ) -> None: ...
    def materialize(
        self,
        shape: tuple[int, ...],
        device: Device = None,
        dtype: dtype | None = None,
    ) -> None: ...

class _BufferMeta(_TensorMeta):
    def __instancecheck__(self, instance: object) -> bool: ...

class Buffer(Tensor, metaclass=_BufferMeta):
    persistent: bool
    def __init__(
        self,
        data: Tensor | None = None,
        *,
        persistent: bool = True,
    ) -> None: ...

class UninitializedBuffer(Tensor):
    persistent: bool
    def __init__(
        self,
        requires_grad: bool = False,
        device: Device = None,
        dtype: dtype | None = None,
        persistent: bool = True,
    ) -> None: ...
    def materialize(
        self,
        shape: tuple[int, ...],
        device: Device = None,
        dtype: dtype | None = None,
    ) -> None: ...
