# In some cases, these basic types are shadowed by corresponding
# top-level values.  The underscore variants let us refer to these
# types.  See https://github.com/python/mypy/issues/4146 for why these
# workarounds is necessary
import os
from builtins import (  # noqa: F401
    bool as _bool,
    bytes as _bytes,
    complex as _complex,
    float as _float,
    int as _int,
    str as _str,
)
from collections.abc import Iterator, Sequence
from typing import (
    Any,
    IO,
    Protocol,
    runtime_checkable,
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
    Union,
)
from typing_extensions import Self

# `as` imports have better static analysis support than assignment `ExposedType: TypeAlias = HiddenType`
from torch import (  # noqa: F401
    device as _device,
    DispatchKey,
    dtype as _dtype,
    layout as _layout,
    qscheme as _qscheme,
    Size,
    SymBool,
    SymFloat,
    SymInt,
    Tensor,
)


if TYPE_CHECKING:
    from torch.autograd.graph import GradientEdge


__all__ = ["Number", "Device", "FileLike", "Storage", "ArrayLike"]

# Convenience aliases for common composite types that we need
# to talk about in PyTorch
_TensorOrTensors: TypeAlias = Tensor | Sequence[Tensor]  # noqa: PYI047
_TensorOrTensorsOrGradEdge: TypeAlias = Union[  # noqa: PYI047
    Tensor,
    Sequence[Tensor],
    "GradientEdge",
    Sequence["GradientEdge"],
]

_size: TypeAlias = Size | list[int] | tuple[int, ...]  # noqa: PYI042,PYI047
_symsize: TypeAlias = Size | Sequence[int | SymInt]  # noqa: PYI042,PYI047
_dispatchkey: TypeAlias = str | DispatchKey  # noqa: PYI042,PYI047

# int or SymInt
IntLikeType: TypeAlias = int | SymInt
# float or SymFloat
FloatLikeType: TypeAlias = float | SymFloat
# bool or SymBool
BoolLikeType: TypeAlias = bool | SymBool

py_sym_types = (SymInt, SymFloat, SymBool)  # left un-annotated intentionally
PySymType: TypeAlias = SymInt | SymFloat | SymBool

# Meta-type for "numeric" things; matches our docs
Number: TypeAlias = int | float | bool
# tuple for isinstance(x, Number) checks.
# FIXME: refactor once python 3.9 support is dropped.
_Number = (int, float, bool)

FileLike: TypeAlias = str | os.PathLike[str] | IO[bytes]

# TypeVar used in Protocols below
_T_co = TypeVar("_T_co", covariant=True)


# Protocol for nested sequences (matches numpy._typing._NestedSequence)
@runtime_checkable
class _NestedSequence(Protocol[_T_co]):
    """A protocol for representing nested sequences of elements."""

    def __len__(self, /) -> int: ...
    def __getitem__(self, index: int, /) -> "_T_co | _NestedSequence[_T_co]": ...
    def __contains__(self, x: object, /) -> bool: ...
    def __iter__(self, /) -> "Iterator[_T_co | _NestedSequence[_T_co]]": ...


# Protocol for objects that support the array protocol (__array__ method)
@runtime_checkable
class _SupportsArray(Protocol):
    """A protocol for objects that can be converted to arrays via __array__."""

    def __array__(self) -> Any: ...


# Type alias for array-like inputs accepted by torch.tensor, torch.as_tensor, etc.
# This includes:
#   - Tensor: PyTorch tensors
#   - numpy.ndarray: NumPy arrays (via import)
#   - _SupportsArray: Objects with __array__ method (e.g., custom array types)
#   - _NestedSequence: Nested lists/tuples of numbers
#   - Scalar types: bool, int, float, complex
#   - bytes/bytearray: For certain dtype conversions
ArrayLike: TypeAlias = Union[
    Tensor,
    "numpy.ndarray[Any, Any]",  # noqa: F821  # pyrefly: ignore[unknown-name]  # numpy is optional
    _SupportsArray,
    _NestedSequence[Union[bool, int, float, complex]],
    bool,
    int,
    float,
    complex,
    bytes,
    bytearray,
]

# Meta-type for "device-like" things.  Not to be confused with 'device' (a
# literal device object).  This nomenclature is consistent with PythonArgParser.
# None means use the default device (typically CPU)
Device: TypeAlias = _device | str | int | None


# Storage protocol implemented by ${Type}StorageBase classes
class Storage:
    _cdata: int
    device: _device
    dtype: _dtype
    _torch_load_uninitialized: bool

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        raise NotImplementedError

    def _new_shared(self, size: int) -> Self:
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

    def share_memory_(self) -> Self:
        raise NotImplementedError

    def nbytes(self) -> int:
        raise NotImplementedError

    def cpu(self) -> Self:
        raise NotImplementedError

    def data_ptr(self) -> int:
        raise NotImplementedError

    def from_file(
        self,
        filename: str,
        shared: bool = False,
        nbytes: int = 0,
    ) -> Self:
        raise NotImplementedError

    def _new_with_file(
        self,
        f: Any,
        element_size: int,
    ) -> Self:
        raise NotImplementedError
