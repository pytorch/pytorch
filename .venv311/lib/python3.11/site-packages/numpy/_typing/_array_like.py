import sys
from collections.abc import Callable, Collection, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeVar, runtime_checkable

import numpy as np
from numpy import dtype

from ._nbit_base import _32Bit, _64Bit
from ._nested_sequence import _NestedSequence
from ._shape import _AnyShape

if TYPE_CHECKING:
    StringDType = np.dtypes.StringDType
else:
    # at runtime outside of type checking importing this from numpy.dtypes
    # would lead to a circular import
    from numpy._core.multiarray import StringDType

_T = TypeVar("_T")
_ScalarT = TypeVar("_ScalarT", bound=np.generic)
_DTypeT = TypeVar("_DTypeT", bound=dtype[Any])
_DTypeT_co = TypeVar("_DTypeT_co", covariant=True, bound=dtype[Any])

NDArray: TypeAlias = np.ndarray[_AnyShape, dtype[_ScalarT]]

# The `_SupportsArray` protocol only cares about the default dtype
# (i.e. `dtype=None` or no `dtype` parameter at all) of the to-be returned
# array.
# Concrete implementations of the protocol are responsible for adding
# any and all remaining overloads
@runtime_checkable
class _SupportsArray(Protocol[_DTypeT_co]):
    def __array__(self) -> np.ndarray[Any, _DTypeT_co]: ...


@runtime_checkable
class _SupportsArrayFunc(Protocol):
    """A protocol class representing `~class.__array_function__`."""
    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Collection[type[Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> object: ...


# TODO: Wait until mypy supports recursive objects in combination with typevars
_FiniteNestedSequence: TypeAlias = (
    _T
    | Sequence[_T]
    | Sequence[Sequence[_T]]
    | Sequence[Sequence[Sequence[_T]]]
    | Sequence[Sequence[Sequence[Sequence[_T]]]]
)

# A subset of `npt.ArrayLike` that can be parametrized w.r.t. `np.generic`
_ArrayLike: TypeAlias = (
    _SupportsArray[dtype[_ScalarT]]
    | _NestedSequence[_SupportsArray[dtype[_ScalarT]]]
)

# A union representing array-like objects; consists of two typevars:
# One representing types that can be parametrized w.r.t. `np.dtype`
# and another one for the rest
_DualArrayLike: TypeAlias = (
    _SupportsArray[_DTypeT]
    | _NestedSequence[_SupportsArray[_DTypeT]]
    | _T
    | _NestedSequence[_T]
)

if sys.version_info >= (3, 12):
    from collections.abc import Buffer as _Buffer
else:
    @runtime_checkable
    class _Buffer(Protocol):
        def __buffer__(self, flags: int, /) -> memoryview: ...

ArrayLike: TypeAlias = _Buffer | _DualArrayLike[dtype[Any], complex | bytes | str]

# `ArrayLike<X>_co`: array-like objects that can be coerced into `X`
# given the casting rules `same_kind`
_ArrayLikeBool_co: TypeAlias = _DualArrayLike[dtype[np.bool], bool]
_ArrayLikeUInt_co: TypeAlias = _DualArrayLike[dtype[np.bool | np.unsignedinteger], bool]
_ArrayLikeInt_co: TypeAlias = _DualArrayLike[dtype[np.bool | np.integer], int]
_ArrayLikeFloat_co: TypeAlias = _DualArrayLike[dtype[np.bool | np.integer | np.floating], float]
_ArrayLikeComplex_co: TypeAlias = _DualArrayLike[dtype[np.bool | np.number], complex]
_ArrayLikeNumber_co: TypeAlias = _ArrayLikeComplex_co
_ArrayLikeTD64_co: TypeAlias = _DualArrayLike[dtype[np.bool | np.integer | np.timedelta64], int]
_ArrayLikeDT64_co: TypeAlias = _ArrayLike[np.datetime64]
_ArrayLikeObject_co: TypeAlias = _ArrayLike[np.object_]

_ArrayLikeVoid_co: TypeAlias = _ArrayLike[np.void]
_ArrayLikeBytes_co: TypeAlias = _DualArrayLike[dtype[np.bytes_], bytes]
_ArrayLikeStr_co: TypeAlias = _DualArrayLike[dtype[np.str_], str]
_ArrayLikeString_co: TypeAlias = _DualArrayLike[StringDType, str]
_ArrayLikeAnyString_co: TypeAlias = _DualArrayLike[dtype[np.character] | StringDType, bytes | str]

__Float64_co: TypeAlias = np.floating[_64Bit] | np.float32 | np.float16 | np.integer | np.bool
__Complex128_co: TypeAlias = np.number[_64Bit] | np.number[_32Bit] | np.float16 | np.integer | np.bool
_ArrayLikeFloat64_co: TypeAlias = _DualArrayLike[dtype[__Float64_co], float]
_ArrayLikeComplex128_co: TypeAlias = _DualArrayLike[dtype[__Complex128_co], complex]

# NOTE: This includes `builtins.bool`, but not `numpy.bool`.
_ArrayLikeInt: TypeAlias = _DualArrayLike[dtype[np.integer], int]
