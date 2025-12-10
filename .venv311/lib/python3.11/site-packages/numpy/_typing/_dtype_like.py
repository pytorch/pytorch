from collections.abc import Sequence  # noqa: F811
from typing import (
    Any,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
    runtime_checkable,
)

import numpy as np

from ._char_codes import (
    _BoolCodes,
    _BytesCodes,
    _ComplexFloatingCodes,
    _DT64Codes,
    _FloatingCodes,
    _NumberCodes,
    _ObjectCodes,
    _SignedIntegerCodes,
    _StrCodes,
    _TD64Codes,
    _UnsignedIntegerCodes,
    _VoidCodes,
)

_ScalarT = TypeVar("_ScalarT", bound=np.generic)
_DTypeT_co = TypeVar("_DTypeT_co", bound=np.dtype, covariant=True)

_DTypeLikeNested: TypeAlias = Any  # TODO: wait for support for recursive types


# Mandatory keys
class _DTypeDictBase(TypedDict):
    names: Sequence[str]
    formats: Sequence[_DTypeLikeNested]


# Mandatory + optional keys
class _DTypeDict(_DTypeDictBase, total=False):
    # Only `str` elements are usable as indexing aliases,
    # but `titles` can in principle accept any object
    offsets: Sequence[int]
    titles: Sequence[Any]
    itemsize: int
    aligned: bool


# A protocol for anything with the dtype attribute
@runtime_checkable
class _SupportsDType(Protocol[_DTypeT_co]):
    @property
    def dtype(self) -> _DTypeT_co: ...


# A subset of `npt.DTypeLike` that can be parametrized w.r.t. `np.generic`
_DTypeLike: TypeAlias = type[_ScalarT] | np.dtype[_ScalarT] | _SupportsDType[np.dtype[_ScalarT]]


# Would create a dtype[np.void]
_VoidDTypeLike: TypeAlias = (
    # If a tuple, then it can be either:
    # - (flexible_dtype, itemsize)
    # - (fixed_dtype, shape)
    # - (base_dtype, new_dtype)
    # But because `_DTypeLikeNested = Any`, the first two cases are redundant

    # tuple[_DTypeLikeNested, int] | tuple[_DTypeLikeNested, _ShapeLike] |
    tuple[_DTypeLikeNested, _DTypeLikeNested]

    # [(field_name, field_dtype, field_shape), ...]
    # The type here is quite broad because NumPy accepts quite a wide
    # range of inputs inside the list; see the tests for some examples.
    | list[Any]

    # {'names': ..., 'formats': ..., 'offsets': ..., 'titles': ..., 'itemsize': ...}
    | _DTypeDict
)

# Aliases for commonly used dtype-like objects.
# Note that the precision of `np.number` subclasses is ignored herein.
_DTypeLikeBool: TypeAlias = type[bool] | _DTypeLike[np.bool] | _BoolCodes
_DTypeLikeInt: TypeAlias = (
    type[int] | _DTypeLike[np.signedinteger] | _SignedIntegerCodes
)
_DTypeLikeUInt: TypeAlias = _DTypeLike[np.unsignedinteger] | _UnsignedIntegerCodes
_DTypeLikeFloat: TypeAlias = type[float] | _DTypeLike[np.floating] | _FloatingCodes
_DTypeLikeComplex: TypeAlias = (
    type[complex] | _DTypeLike[np.complexfloating] | _ComplexFloatingCodes
)
_DTypeLikeComplex_co: TypeAlias = (
    type[complex] | _DTypeLike[np.bool | np.number] | _BoolCodes | _NumberCodes
)
_DTypeLikeDT64: TypeAlias = _DTypeLike[np.timedelta64] | _TD64Codes
_DTypeLikeTD64: TypeAlias = _DTypeLike[np.datetime64] | _DT64Codes
_DTypeLikeBytes: TypeAlias = type[bytes] | _DTypeLike[np.bytes_] | _BytesCodes
_DTypeLikeStr: TypeAlias = type[str] | _DTypeLike[np.str_] | _StrCodes
_DTypeLikeVoid: TypeAlias = (
    type[memoryview] | _DTypeLike[np.void] | _VoidDTypeLike | _VoidCodes
)
_DTypeLikeObject: TypeAlias = type[object] | _DTypeLike[np.object_] | _ObjectCodes


# Anything that can be coerced into numpy.dtype.
# Reference: https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
DTypeLike: TypeAlias = _DTypeLike[Any] | _VoidDTypeLike | str | None

# NOTE: while it is possible to provide the dtype as a dict of
# dtype-like objects (e.g. `{'field1': ..., 'field2': ..., ...}`),
# this syntax is officially discouraged and
# therefore not included in the type-union defining `DTypeLike`.
#
# See https://github.com/numpy/numpy/issues/16891 for more details.
