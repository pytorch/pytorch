from typing import Any, Generic, NamedTuple, SupportsIndex, TypeAlias, overload
from typing import Literal as L

from typing_extensions import TypeVar, deprecated

import numpy as np
from numpy._typing import (
    ArrayLike,
    NDArray,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeNumber_co,
)

__all__ = [
    "ediff1d",
    "in1d",
    "intersect1d",
    "isin",
    "setdiff1d",
    "setxor1d",
    "union1d",
    "unique",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
]

_ScalarT = TypeVar("_ScalarT", bound=np.generic)
_NumericT = TypeVar("_NumericT", bound=np.number | np.timedelta64 | np.object_)

# Explicitly set all allowed values to prevent accidental castings to
# abstract dtypes (their common super-type).
# Only relevant if two or more arguments are parametrized, (e.g. `setdiff1d`)
# which could result in, for example, `int64` and `float64`producing a
# `number[_64Bit]` array
_EitherSCT = TypeVar(
    "_EitherSCT",
    np.bool,
    np.int8, np.int16, np.int32, np.int64, np.intp,
    np.uint8, np.uint16, np.uint32, np.uint64, np.uintp,
    np.float16, np.float32, np.float64, np.longdouble,
    np.complex64, np.complex128, np.clongdouble,
    np.timedelta64, np.datetime64,
    np.bytes_, np.str_, np.void, np.object_,
    np.integer, np.floating, np.complexfloating, np.character,
)  # fmt: skip

_AnyArray: TypeAlias = NDArray[Any]
_IntArray: TypeAlias = NDArray[np.intp]

###

class UniqueAllResult(NamedTuple, Generic[_ScalarT]):
    values: NDArray[_ScalarT]
    indices: _IntArray
    inverse_indices: _IntArray
    counts: _IntArray

class UniqueCountsResult(NamedTuple, Generic[_ScalarT]):
    values: NDArray[_ScalarT]
    counts: _IntArray

class UniqueInverseResult(NamedTuple, Generic[_ScalarT]):
    values: NDArray[_ScalarT]
    inverse_indices: _IntArray

#
@overload
def ediff1d(
    ary: _ArrayLikeBool_co,
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> NDArray[np.int8]: ...
@overload
def ediff1d(
    ary: _ArrayLike[_NumericT],
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> NDArray[_NumericT]: ...
@overload
def ediff1d(
    ary: _ArrayLike[np.datetime64[Any]],
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> NDArray[np.timedelta64]: ...
@overload
def ediff1d(
    ary: _ArrayLikeNumber_co,
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> _AnyArray: ...

#
@overload  # known scalar-type, FFF
def unique(
    ar: _ArrayLike[_ScalarT],
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> NDArray[_ScalarT]: ...
@overload  # unknown scalar-type, FFF
def unique(
    ar: ArrayLike,
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> _AnyArray: ...
@overload  # known scalar-type, TFF
def unique(
    ar: _ArrayLike[_ScalarT],
    return_index: L[True],
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[_ScalarT], _IntArray]: ...
@overload  # unknown scalar-type, TFF
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[False] = False,
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_AnyArray, _IntArray]: ...
@overload  # known scalar-type, FTF (positional)
def unique(
    ar: _ArrayLike[_ScalarT],
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[_ScalarT], _IntArray]: ...
@overload  # known scalar-type, FTF (keyword)
def unique(
    ar: _ArrayLike[_ScalarT],
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[_ScalarT], _IntArray]: ...
@overload  # unknown scalar-type, FTF (positional)
def unique(
    ar: ArrayLike,
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_AnyArray, _IntArray]: ...
@overload  # unknown scalar-type, FTF (keyword)
def unique(
    ar: ArrayLike,
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_AnyArray, _IntArray]: ...
@overload  # known scalar-type, FFT (positional)
def unique(
    ar: _ArrayLike[_ScalarT],
    return_index: L[False],
    return_inverse: L[False],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[_ScalarT], _IntArray]: ...
@overload  # known scalar-type, FFT (keyword)
def unique(
    ar: _ArrayLike[_ScalarT],
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[_ScalarT], _IntArray]: ...
@overload  # unknown scalar-type, FFT (positional)
def unique(
    ar: ArrayLike,
    return_index: L[False],
    return_inverse: L[False],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_AnyArray, _IntArray]: ...
@overload  # unknown scalar-type, FFT (keyword)
def unique(
    ar: ArrayLike,
    return_index: L[False] = False,
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_AnyArray, _IntArray]: ...
@overload  # known scalar-type, TTF
def unique(
    ar: _ArrayLike[_ScalarT],
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[_ScalarT], _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, TTF
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[False] = False,
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_AnyArray, _IntArray, _IntArray]: ...
@overload  # known scalar-type, TFT (positional)
def unique(
    ar: _ArrayLike[_ScalarT],
    return_index: L[True],
    return_inverse: L[False],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[_ScalarT], _IntArray, _IntArray]: ...
@overload  # known scalar-type, TFT (keyword)
def unique(
    ar: _ArrayLike[_ScalarT],
    return_index: L[True],
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[_ScalarT], _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, TFT (positional)
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[False],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_AnyArray, _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, TFT (keyword)
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[False] = False,
    *,
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_AnyArray, _IntArray, _IntArray]: ...
@overload  # known scalar-type, FTT (positional)
def unique(
    ar: _ArrayLike[_ScalarT],
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[_ScalarT], _IntArray, _IntArray]: ...
@overload  # known scalar-type, FTT (keyword)
def unique(
    ar: _ArrayLike[_ScalarT],
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[_ScalarT], _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, FTT (positional)
def unique(
    ar: ArrayLike,
    return_index: L[False],
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_AnyArray, _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, FTT (keyword)
def unique(
    ar: ArrayLike,
    return_index: L[False] = False,
    *,
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_AnyArray, _IntArray, _IntArray]: ...
@overload  # known scalar-type, TTT
def unique(
    ar: _ArrayLike[_ScalarT],
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[NDArray[_ScalarT], _IntArray, _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, TTT
def unique(
    ar: ArrayLike,
    return_index: L[True],
    return_inverse: L[True],
    return_counts: L[True],
    axis: SupportsIndex | None = None,
    *,
    equal_nan: bool = True,
    sorted: bool = True,
) -> tuple[_AnyArray, _IntArray, _IntArray, _IntArray]: ...

#
@overload
def unique_all(x: _ArrayLike[_ScalarT]) -> UniqueAllResult[_ScalarT]: ...
@overload
def unique_all(x: ArrayLike) -> UniqueAllResult[Any]: ...

#
@overload
def unique_counts(x: _ArrayLike[_ScalarT]) -> UniqueCountsResult[_ScalarT]: ...
@overload
def unique_counts(x: ArrayLike) -> UniqueCountsResult[Any]: ...

#
@overload
def unique_inverse(x: _ArrayLike[_ScalarT]) -> UniqueInverseResult[_ScalarT]: ...
@overload
def unique_inverse(x: ArrayLike) -> UniqueInverseResult[Any]: ...

#
@overload
def unique_values(x: _ArrayLike[_ScalarT]) -> NDArray[_ScalarT]: ...
@overload
def unique_values(x: ArrayLike) -> _AnyArray: ...

#
@overload  # known scalar-type, return_indices=False (default)
def intersect1d(
    ar1: _ArrayLike[_EitherSCT],
    ar2: _ArrayLike[_EitherSCT],
    assume_unique: bool = False,
    return_indices: L[False] = False,
) -> NDArray[_EitherSCT]: ...
@overload  # known scalar-type, return_indices=True (positional)
def intersect1d(
    ar1: _ArrayLike[_EitherSCT],
    ar2: _ArrayLike[_EitherSCT],
    assume_unique: bool,
    return_indices: L[True],
) -> tuple[NDArray[_EitherSCT], _IntArray, _IntArray]: ...
@overload  # known scalar-type, return_indices=True (keyword)
def intersect1d(
    ar1: _ArrayLike[_EitherSCT],
    ar2: _ArrayLike[_EitherSCT],
    assume_unique: bool = False,
    *,
    return_indices: L[True],
) -> tuple[NDArray[_EitherSCT], _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, return_indices=False (default)
def intersect1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: bool = False,
    return_indices: L[False] = False,
) -> _AnyArray: ...
@overload  # unknown scalar-type, return_indices=True (positional)
def intersect1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: bool,
    return_indices: L[True],
) -> tuple[_AnyArray, _IntArray, _IntArray]: ...
@overload  # unknown scalar-type, return_indices=True (keyword)
def intersect1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: bool = False,
    *,
    return_indices: L[True],
) -> tuple[_AnyArray, _IntArray, _IntArray]: ...

#
@overload
def setxor1d(ar1: _ArrayLike[_EitherSCT], ar2: _ArrayLike[_EitherSCT], assume_unique: bool = False) -> NDArray[_EitherSCT]: ...
@overload
def setxor1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False) -> _AnyArray: ...

#
@overload
def union1d(ar1: _ArrayLike[_EitherSCT], ar2: _ArrayLike[_EitherSCT]) -> NDArray[_EitherSCT]: ...
@overload
def union1d(ar1: ArrayLike, ar2: ArrayLike) -> _AnyArray: ...

#
@overload
def setdiff1d(ar1: _ArrayLike[_EitherSCT], ar2: _ArrayLike[_EitherSCT], assume_unique: bool = False) -> NDArray[_EitherSCT]: ...
@overload
def setdiff1d(ar1: ArrayLike, ar2: ArrayLike, assume_unique: bool = False) -> _AnyArray: ...

#
def isin(
    element: ArrayLike,
    test_elements: ArrayLike,
    assume_unique: bool = False,
    invert: bool = False,
    *,
    kind: L["sort", "table"] | None = None,
) -> NDArray[np.bool]: ...

#
@deprecated("Use 'isin' instead")
def in1d(
    element: ArrayLike,
    test_elements: ArrayLike,
    assume_unique: bool = False,
    invert: bool = False,
    *,
    kind: L["sort", "table"] | None = None,
) -> NDArray[np.bool]: ...
