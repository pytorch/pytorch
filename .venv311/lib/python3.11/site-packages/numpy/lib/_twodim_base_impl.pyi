from collections.abc import Callable, Sequence
from typing import (
    Any,
    TypeAlias,
    TypeVar,
    overload,
)
from typing import (
    Literal as L,
)

import numpy as np
from numpy import (
    _OrderCF,
    complex128,
    complexfloating,
    datetime64,
    float64,
    floating,
    generic,
    int_,
    intp,
    object_,
    signedinteger,
    timedelta64,
)
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _ArrayLike,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _DTypeLike,
    _SupportsArray,
    _SupportsArrayFunc,
)

__all__ = [
    "diag",
    "diagflat",
    "eye",
    "fliplr",
    "flipud",
    "tri",
    "triu",
    "tril",
    "vander",
    "histogram2d",
    "mask_indices",
    "tril_indices",
    "tril_indices_from",
    "triu_indices",
    "triu_indices_from",
]

###

_T = TypeVar("_T")
_ScalarT = TypeVar("_ScalarT", bound=generic)
_ComplexFloatingT = TypeVar("_ComplexFloatingT", bound=np.complexfloating)
_InexactT = TypeVar("_InexactT", bound=np.inexact)
_NumberCoT = TypeVar("_NumberCoT", bound=_Number_co)

# The returned arrays dtype must be compatible with `np.equal`
_MaskFunc: TypeAlias = Callable[[NDArray[int_], _T], NDArray[_Number_co | timedelta64 | datetime64 | object_]]

_Int_co: TypeAlias = np.integer | np.bool
_Float_co: TypeAlias = np.floating | _Int_co
_Number_co: TypeAlias = np.number | np.bool

_ArrayLike1D: TypeAlias = _SupportsArray[np.dtype[_ScalarT]] | Sequence[_ScalarT]
_ArrayLike1DInt_co: TypeAlias = _SupportsArray[np.dtype[_Int_co]] | Sequence[int | _Int_co]
_ArrayLike1DFloat_co: TypeAlias = _SupportsArray[np.dtype[_Float_co]] | Sequence[float | _Float_co]
_ArrayLike2DFloat_co: TypeAlias = _SupportsArray[np.dtype[_Float_co]] | Sequence[_ArrayLike1DFloat_co]
_ArrayLike1DNumber_co: TypeAlias = _SupportsArray[np.dtype[_Number_co]] | Sequence[complex | _Number_co]

###

@overload
def fliplr(m: _ArrayLike[_ScalarT]) -> NDArray[_ScalarT]: ...
@overload
def fliplr(m: ArrayLike) -> NDArray[Any]: ...

@overload
def flipud(m: _ArrayLike[_ScalarT]) -> NDArray[_ScalarT]: ...
@overload
def flipud(m: ArrayLike) -> NDArray[Any]: ...

@overload
def eye(
    N: int,
    M: int | None = ...,
    k: int = ...,
    dtype: None = ...,
    order: _OrderCF = ...,
    *,
    device: L["cpu"] | None = ...,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[float64]: ...
@overload
def eye(
    N: int,
    M: int | None,
    k: int,
    dtype: _DTypeLike[_ScalarT],
    order: _OrderCF = ...,
    *,
    device: L["cpu"] | None = ...,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[_ScalarT]: ...
@overload
def eye(
    N: int,
    M: int | None = ...,
    k: int = ...,
    *,
    dtype: _DTypeLike[_ScalarT],
    order: _OrderCF = ...,
    device: L["cpu"] | None = ...,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[_ScalarT]: ...
@overload
def eye(
    N: int,
    M: int | None = ...,
    k: int = ...,
    dtype: DTypeLike = ...,
    order: _OrderCF = ...,
    *,
    device: L["cpu"] | None = ...,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[Any]: ...

@overload
def diag(v: _ArrayLike[_ScalarT], k: int = ...) -> NDArray[_ScalarT]: ...
@overload
def diag(v: ArrayLike, k: int = ...) -> NDArray[Any]: ...

@overload
def diagflat(v: _ArrayLike[_ScalarT], k: int = ...) -> NDArray[_ScalarT]: ...
@overload
def diagflat(v: ArrayLike, k: int = ...) -> NDArray[Any]: ...

@overload
def tri(
    N: int,
    M: int | None = ...,
    k: int = ...,
    dtype: None = ...,
    *,
    like: _SupportsArrayFunc | None = ...
) -> NDArray[float64]: ...
@overload
def tri(
    N: int,
    M: int | None,
    k: int,
    dtype: _DTypeLike[_ScalarT],
    *,
    like: _SupportsArrayFunc | None = ...
) -> NDArray[_ScalarT]: ...
@overload
def tri(
    N: int,
    M: int | None = ...,
    k: int = ...,
    *,
    dtype: _DTypeLike[_ScalarT],
    like: _SupportsArrayFunc | None = ...
) -> NDArray[_ScalarT]: ...
@overload
def tri(
    N: int,
    M: int | None = ...,
    k: int = ...,
    dtype: DTypeLike = ...,
    *,
    like: _SupportsArrayFunc | None = ...
) -> NDArray[Any]: ...

@overload
def tril(m: _ArrayLike[_ScalarT], k: int = 0) -> NDArray[_ScalarT]: ...
@overload
def tril(m: ArrayLike, k: int = 0) -> NDArray[Any]: ...

@overload
def triu(m: _ArrayLike[_ScalarT], k: int = 0) -> NDArray[_ScalarT]: ...
@overload
def triu(m: ArrayLike, k: int = 0) -> NDArray[Any]: ...

@overload
def vander(  # type: ignore[misc]
    x: _ArrayLikeInt_co,
    N: int | None = ...,
    increasing: bool = ...,
) -> NDArray[signedinteger]: ...
@overload
def vander(  # type: ignore[misc]
    x: _ArrayLikeFloat_co,
    N: int | None = ...,
    increasing: bool = ...,
) -> NDArray[floating]: ...
@overload
def vander(
    x: _ArrayLikeComplex_co,
    N: int | None = ...,
    increasing: bool = ...,
) -> NDArray[complexfloating]: ...
@overload
def vander(
    x: _ArrayLikeObject_co,
    N: int | None = ...,
    increasing: bool = ...,
) -> NDArray[object_]: ...

@overload
def histogram2d(
    x: _ArrayLike1D[_ComplexFloatingT],
    y: _ArrayLike1D[_ComplexFloatingT | _Float_co],
    bins: int | Sequence[int] = ...,
    range: _ArrayLike2DFloat_co | None = ...,
    density: bool | None = ...,
    weights: _ArrayLike1DFloat_co | None = ...,
) -> tuple[
    NDArray[float64],
    NDArray[_ComplexFloatingT],
    NDArray[_ComplexFloatingT],
]: ...
@overload
def histogram2d(
    x: _ArrayLike1D[_ComplexFloatingT | _Float_co],
    y: _ArrayLike1D[_ComplexFloatingT],
    bins: int | Sequence[int] = ...,
    range: _ArrayLike2DFloat_co | None = ...,
    density: bool | None = ...,
    weights: _ArrayLike1DFloat_co | None = ...,
) -> tuple[
    NDArray[float64],
    NDArray[_ComplexFloatingT],
    NDArray[_ComplexFloatingT],
]: ...
@overload
def histogram2d(
    x: _ArrayLike1D[_InexactT],
    y: _ArrayLike1D[_InexactT | _Int_co],
    bins: int | Sequence[int] = ...,
    range: _ArrayLike2DFloat_co | None = ...,
    density: bool | None = ...,
    weights: _ArrayLike1DFloat_co | None = ...,
) -> tuple[
    NDArray[float64],
    NDArray[_InexactT],
    NDArray[_InexactT],
]: ...
@overload
def histogram2d(
    x: _ArrayLike1D[_InexactT | _Int_co],
    y: _ArrayLike1D[_InexactT],
    bins: int | Sequence[int] = ...,
    range: _ArrayLike2DFloat_co | None = ...,
    density: bool | None = ...,
    weights: _ArrayLike1DFloat_co | None = ...,
) -> tuple[
    NDArray[float64],
    NDArray[_InexactT],
    NDArray[_InexactT],
]: ...
@overload
def histogram2d(
    x: _ArrayLike1DInt_co | Sequence[float],
    y: _ArrayLike1DInt_co | Sequence[float],
    bins: int | Sequence[int] = ...,
    range: _ArrayLike2DFloat_co | None = ...,
    density: bool | None = ...,
    weights: _ArrayLike1DFloat_co | None = ...,
) -> tuple[
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
]: ...
@overload
def histogram2d(
    x: Sequence[complex],
    y: Sequence[complex],
    bins: int | Sequence[int] = ...,
    range: _ArrayLike2DFloat_co | None = ...,
    density: bool | None = ...,
    weights: _ArrayLike1DFloat_co | None = ...,
) -> tuple[
    NDArray[float64],
    NDArray[complex128 | float64],
    NDArray[complex128 | float64],
]: ...
@overload
def histogram2d(
    x: _ArrayLike1DNumber_co,
    y: _ArrayLike1DNumber_co,
    bins: _ArrayLike1D[_NumberCoT] | Sequence[_ArrayLike1D[_NumberCoT]],
    range: _ArrayLike2DFloat_co | None = ...,
    density: bool | None = ...,
    weights: _ArrayLike1DFloat_co | None = ...,
) -> tuple[
    NDArray[float64],
    NDArray[_NumberCoT],
    NDArray[_NumberCoT],
]: ...
@overload
def histogram2d(
    x: _ArrayLike1D[_InexactT],
    y: _ArrayLike1D[_InexactT],
    bins: Sequence[_ArrayLike1D[_NumberCoT] | int],
    range: _ArrayLike2DFloat_co | None = ...,
    density: bool | None = ...,
    weights: _ArrayLike1DFloat_co | None = ...,
) -> tuple[
    NDArray[float64],
    NDArray[_NumberCoT | _InexactT],
    NDArray[_NumberCoT | _InexactT],
]: ...
@overload
def histogram2d(
    x: _ArrayLike1DInt_co | Sequence[float],
    y: _ArrayLike1DInt_co | Sequence[float],
    bins: Sequence[_ArrayLike1D[_NumberCoT] | int],
    range: _ArrayLike2DFloat_co | None = ...,
    density: bool | None = ...,
    weights: _ArrayLike1DFloat_co | None = ...,
) -> tuple[
    NDArray[float64],
    NDArray[_NumberCoT | float64],
    NDArray[_NumberCoT | float64],
]: ...
@overload
def histogram2d(
    x: Sequence[complex],
    y: Sequence[complex],
    bins: Sequence[_ArrayLike1D[_NumberCoT] | int],
    range: _ArrayLike2DFloat_co | None = ...,
    density: bool | None = ...,
    weights: _ArrayLike1DFloat_co | None = ...,
) -> tuple[
    NDArray[float64],
    NDArray[_NumberCoT | complex128 | float64],
    NDArray[_NumberCoT | complex128 | float64],
]: ...
@overload
def histogram2d(
    x: _ArrayLike1DNumber_co,
    y: _ArrayLike1DNumber_co,
    bins: Sequence[Sequence[bool]],
    range: _ArrayLike2DFloat_co | None = ...,
    density: bool | None = ...,
    weights: _ArrayLike1DFloat_co | None = ...,
) -> tuple[
    NDArray[float64],
    NDArray[np.bool],
    NDArray[np.bool],
]: ...
@overload
def histogram2d(
    x: _ArrayLike1DNumber_co,
    y: _ArrayLike1DNumber_co,
    bins: Sequence[Sequence[int]],
    range: _ArrayLike2DFloat_co | None = ...,
    density: bool | None = ...,
    weights: _ArrayLike1DFloat_co | None = ...,
) -> tuple[
    NDArray[float64],
    NDArray[np.int_ | np.bool],
    NDArray[np.int_ | np.bool],
]: ...
@overload
def histogram2d(
    x: _ArrayLike1DNumber_co,
    y: _ArrayLike1DNumber_co,
    bins: Sequence[Sequence[float]],
    range: _ArrayLike2DFloat_co | None = ...,
    density: bool | None = ...,
    weights: _ArrayLike1DFloat_co | None = ...,
) -> tuple[
    NDArray[float64],
    NDArray[np.float64 | np.int_ | np.bool],
    NDArray[np.float64 | np.int_ | np.bool],
]: ...
@overload
def histogram2d(
    x: _ArrayLike1DNumber_co,
    y: _ArrayLike1DNumber_co,
    bins: Sequence[Sequence[complex]],
    range: _ArrayLike2DFloat_co | None = ...,
    density: bool | None = ...,
    weights: _ArrayLike1DFloat_co | None = ...,
) -> tuple[
    NDArray[float64],
    NDArray[np.complex128 | np.float64 | np.int_ | np.bool],
    NDArray[np.complex128 | np.float64 | np.int_ | np.bool],
]: ...

# NOTE: we're assuming/demanding here the `mask_func` returns
# an ndarray of shape `(n, n)`; otherwise there is the possibility
# of the output tuple having more or less than 2 elements
@overload
def mask_indices(
    n: int,
    mask_func: _MaskFunc[int],
    k: int = ...,
) -> tuple[NDArray[intp], NDArray[intp]]: ...
@overload
def mask_indices(
    n: int,
    mask_func: _MaskFunc[_T],
    k: _T,
) -> tuple[NDArray[intp], NDArray[intp]]: ...

def tril_indices(
    n: int,
    k: int = ...,
    m: int | None = ...,
) -> tuple[NDArray[int_], NDArray[int_]]: ...

def tril_indices_from(
    arr: NDArray[Any],
    k: int = ...,
) -> tuple[NDArray[int_], NDArray[int_]]: ...

def triu_indices(
    n: int,
    k: int = ...,
    m: int | None = ...,
) -> tuple[NDArray[int_], NDArray[int_]]: ...

def triu_indices_from(
    arr: NDArray[Any],
    k: int = ...,
) -> tuple[NDArray[int_], NDArray[int_]]: ...
