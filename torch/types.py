# In some cases, these basic types are shadowed by corresponding
# top-level values.  The underscore variants let us refer to these
# types.  See https://github.com/python/mypy/issues/4146 for why these
# workarounds is necessary
from builtins import (  # noqa: F401
    bool as _bool,
    bytes as _bytes,
    complex as _complex,
    float as _float,
    int as _int,
    str as _str,
)
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING, Union
from typing_extensions import TypeAlias

# `as` imports have better static analysis support than assignment `ExposedType: TypeAlias = HiddenType`
from torch import (
    device as _device,
    DispatchKey,
    dtype as _dtype,
    Size,
    SymBool,
    SymFloat,
    SymInt,
    Tensor,
)


if TYPE_CHECKING:
    from torch.autograd.graph import GradientEdge

__all__ = [
    "Number",
    "Device",
    "Size",
    "SymBool",
    "SymFloat",
    "SymInt",
    "Tensor",
    "Storage",
    "IntLikeType",
    "FloatLikeType",
    "BoolLikeType",
]

# Convenience aliases for common composite types that we need
# to talk about in PyTorch
_TensorOrTensors: TypeAlias = Union[Tensor, Sequence[Tensor]]  # noqa: PYI047
_TensorOrTensorsOrGradEdge: TypeAlias = Union[  # noqa: PYI047
    Tensor,
    Sequence[Tensor],
    "GradientEdge",
    Sequence["GradientEdge"],
]

_size: TypeAlias = Union[Size, list[int], tuple[int, ...]]  # noqa: PYI042,PYI047
_symsize: TypeAlias = Union[Size, Sequence[Union[int, SymInt]]]  # noqa: PYI042,PYI047
_dispatchkey: TypeAlias = Union[str, DispatchKey]  # noqa: PYI042,PYI047

# int or SymInt
IntLikeType: TypeAlias = Union[int, SymInt]
# float or SymFloat
FloatLikeType: TypeAlias = Union[float, SymFloat]
# bool or SymBool
BoolLikeType: TypeAlias = Union[bool, SymBool]

py_sym_types: tuple = (SymInt, SymFloat, SymBool)
PySymType: TypeAlias = Union[SymInt, SymFloat, SymBool]

# Meta-type for "numeric" things; matches our docs
Number: TypeAlias = Union[int, float, bool]

# Meta-type for "device-like" things.  Not to be confused with 'device' (a
# literal device object).  This nomenclature is consistent with PythonArgParser.
# None means use the default device (typically CPU)
Device: TypeAlias = Union[_device, str, int, None]


# Storage protocol implemented by ${Type}StorageBase classes
class Storage:
    _cdata: int
    device: _device
    dtype: _dtype
    _torch_load_uninitialized: bool

    def __deepcopy__(self, memo: dict[int, Any]) -> "Storage":
        raise NotImplementedError

    def _new_shared(self, size: int) -> "Storage":
        raise NotImplementedError

    def _write_file(
        self,
        f: Any,
        is_real_file: bool,
        save_size: bool,
        element_size: int,
    ) -> None:
        raise NotImplementedError

    def element_size(self) -> int:
        raise NotImplementedError

    def is_shared(self) -> bool:
        raise NotImplementedError

    def share_memory_(self) -> "Storage":
        raise NotImplementedError

    def nbytes(self) -> int:
        raise NotImplementedError

    def cpu(self) -> "Storage":
        raise NotImplementedError

    def data_ptr(self) -> int:
        raise NotImplementedError

    def from_file(
        self,
        filename: str,
        shared: bool = False,
        nbytes: int = 0,
    ) -> "Storage":
        raise NotImplementedError

    def _new_with_file(
        self,
        f: Any,
        element_size: int,
    ) -> "Storage":
        raise NotImplementedError
