from typing_extensions import TypeIs

from torch import dtype, Tensor
from torch._C import _TensorMeta
from torch.types import Device

# black magic to set the __module__ attribute for the following classes and functions
__name__: str = "torch.nn.parameter"  # type: ignore[no-redef]

class _ParameterMeta(_TensorMeta):
    def __instancecheck__(self, instance: object) -> bool: ...

class Parameter(Tensor, metaclass=_ParameterMeta):
    def __init__(
        self,
        data: Tensor | None = None,
        requires_grad: bool = True,
    ) -> None: ...

def is_lazy(
    param: Tensor,
) -> TypeIs[UninitializedParameter | UninitializedBuffer]: ...

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
