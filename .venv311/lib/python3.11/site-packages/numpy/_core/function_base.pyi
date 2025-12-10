from typing import Literal as L
from typing import SupportsIndex, TypeAlias, TypeVar, overload

from _typeshed import Incomplete

import numpy as np
from numpy._typing import (
    DTypeLike,
    NDArray,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _DTypeLike,
)
from numpy._typing._array_like import _DualArrayLike

__all__ = ["geomspace", "linspace", "logspace"]

_ScalarT = TypeVar("_ScalarT", bound=np.generic)

_ToArrayFloat64: TypeAlias = _DualArrayLike[np.dtype[np.float64 | np.integer | np.bool], float]

@overload
def linspace(
    start: _ToArrayFloat64,
    stop: _ToArrayFloat64,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    retstep: L[False] = False,
    dtype: None = None,
    axis: SupportsIndex = 0,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[np.float64]: ...
@overload
def linspace(
    start: _ArrayLikeFloat_co,
    stop: _ArrayLikeFloat_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    retstep: L[False] = False,
    dtype: None = None,
    axis: SupportsIndex = 0,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[np.floating]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    retstep: L[False] = False,
    dtype: None = None,
    axis: SupportsIndex = 0,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[np.complexfloating]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex,
    endpoint: bool,
    retstep: L[False],
    dtype: _DTypeLike[_ScalarT],
    axis: SupportsIndex = 0,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[_ScalarT]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    retstep: L[False] = False,
    *,
    dtype: _DTypeLike[_ScalarT],
    axis: SupportsIndex = 0,
    device: L["cpu"] | None = None,
) -> NDArray[_ScalarT]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    retstep: L[False] = False,
    dtype: DTypeLike | None = None,
    axis: SupportsIndex = 0,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[Incomplete]: ...
@overload
def linspace(
    start: _ToArrayFloat64,
    stop: _ToArrayFloat64,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    *,
    retstep: L[True],
    dtype: None = None,
    axis: SupportsIndex = 0,
    device: L["cpu"] | None = None,
) -> tuple[NDArray[np.float64], np.float64]: ...
@overload
def linspace(
    start: _ArrayLikeFloat_co,
    stop: _ArrayLikeFloat_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    *,
    retstep: L[True],
    dtype: None = None,
    axis: SupportsIndex = 0,
    device: L["cpu"] | None = None,
) -> tuple[NDArray[np.floating], np.floating]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    *,
    retstep: L[True],
    dtype: None = None,
    axis: SupportsIndex = 0,
    device: L["cpu"] | None = None,
) -> tuple[NDArray[np.complexfloating], np.complexfloating]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    *,
    retstep: L[True],
    dtype: _DTypeLike[_ScalarT],
    axis: SupportsIndex = 0,
    device: L["cpu"] | None = None,
) -> tuple[NDArray[_ScalarT], _ScalarT]: ...
@overload
def linspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    *,
    retstep: L[True],
    dtype: DTypeLike | None = None,
    axis: SupportsIndex = 0,
    device: L["cpu"] | None = None,
) -> tuple[NDArray[Incomplete], Incomplete]: ...

@overload
def logspace(
    start: _ToArrayFloat64,
    stop: _ToArrayFloat64,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    base: _ToArrayFloat64 = 10.0,
    dtype: None = None,
    axis: SupportsIndex = 0,
) -> NDArray[np.float64]: ...
@overload
def logspace(
    start: _ArrayLikeFloat_co,
    stop: _ArrayLikeFloat_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    base: _ArrayLikeFloat_co = 10.0,
    dtype: None = None,
    axis: SupportsIndex = 0,
) -> NDArray[np.floating]: ...
@overload
def logspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    base: _ArrayLikeComplex_co = 10.0,
    dtype: None = None,
    axis: SupportsIndex = 0,
) -> NDArray[np.complexfloating]: ...
@overload
def logspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex,
    endpoint: bool,
    base: _ArrayLikeComplex_co,
    dtype: _DTypeLike[_ScalarT],
    axis: SupportsIndex = 0,
) -> NDArray[_ScalarT]: ...
@overload
def logspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    base: _ArrayLikeComplex_co = 10.0,
    *,
    dtype: _DTypeLike[_ScalarT],
    axis: SupportsIndex = 0,
) -> NDArray[_ScalarT]: ...
@overload
def logspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    base: _ArrayLikeComplex_co = 10.0,
    dtype: DTypeLike | None = None,
    axis: SupportsIndex = 0,
) -> NDArray[Incomplete]: ...

@overload
def geomspace(
    start: _ToArrayFloat64,
    stop: _ToArrayFloat64,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    dtype: None = None,
    axis: SupportsIndex = 0,
) -> NDArray[np.float64]: ...
@overload
def geomspace(
    start: _ArrayLikeFloat_co,
    stop: _ArrayLikeFloat_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    dtype: None = None,
    axis: SupportsIndex = 0,
) -> NDArray[np.floating]: ...
@overload
def geomspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    dtype: None = None,
    axis: SupportsIndex = 0,
) -> NDArray[np.complexfloating]: ...
@overload
def geomspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex,
    endpoint: bool,
    dtype: _DTypeLike[_ScalarT],
    axis: SupportsIndex = 0,
) -> NDArray[_ScalarT]: ...
@overload
def geomspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    *,
    dtype: _DTypeLike[_ScalarT],
    axis: SupportsIndex = 0,
) -> NDArray[_ScalarT]: ...
@overload
def geomspace(
    start: _ArrayLikeComplex_co,
    stop: _ArrayLikeComplex_co,
    num: SupportsIndex = 50,
    endpoint: bool = True,
    dtype: DTypeLike | None = None,
    axis: SupportsIndex = 0,
) -> NDArray[Incomplete]: ...

def add_newdoc(
    place: str,
    obj: str,
    doc: str | tuple[str, str] | list[tuple[str, str]],
    warn_on_python: bool = True,
) -> None: ...
