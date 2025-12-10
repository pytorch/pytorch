from typing import Any, TypeVar, overload

import numpy as np
from numpy import floating, object_
from numpy._typing import (
    NDArray,
    _ArrayLikeFloat_co,
    _ArrayLikeObject_co,
    _FloatLike_co,
)

__all__ = ["fix", "isneginf", "isposinf"]

_ArrayT = TypeVar("_ArrayT", bound=NDArray[Any])

@overload
def fix(  # type: ignore[misc]
    x: _FloatLike_co,
    out: None = ...,
) -> floating: ...
@overload
def fix(
    x: _ArrayLikeFloat_co,
    out: None = ...,
) -> NDArray[floating]: ...
@overload
def fix(
    x: _ArrayLikeObject_co,
    out: None = ...,
) -> NDArray[object_]: ...
@overload
def fix(
    x: _ArrayLikeFloat_co | _ArrayLikeObject_co,
    out: _ArrayT,
) -> _ArrayT: ...

@overload
def isposinf(  # type: ignore[misc]
    x: _FloatLike_co,
    out: None = ...,
) -> np.bool: ...
@overload
def isposinf(
    x: _ArrayLikeFloat_co,
    out: None = ...,
) -> NDArray[np.bool]: ...
@overload
def isposinf(
    x: _ArrayLikeFloat_co,
    out: _ArrayT,
) -> _ArrayT: ...

@overload
def isneginf(  # type: ignore[misc]
    x: _FloatLike_co,
    out: None = ...,
) -> np.bool: ...
@overload
def isneginf(
    x: _ArrayLikeFloat_co,
    out: None = ...,
) -> NDArray[np.bool]: ...
@overload
def isneginf(
    x: _ArrayLikeFloat_co,
    out: _ArrayT,
) -> _ArrayT: ...
