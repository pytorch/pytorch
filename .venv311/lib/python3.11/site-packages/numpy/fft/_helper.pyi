from typing import Any, Final, TypeVar, overload
from typing import Literal as L

from numpy import complexfloating, floating, generic, integer
from numpy._typing import (
    ArrayLike,
    NDArray,
    _ArrayLike,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ShapeLike,
)

__all__ = ["fftfreq", "fftshift", "ifftshift", "rfftfreq"]

_ScalarT = TypeVar("_ScalarT", bound=generic)

###

integer_types: Final[tuple[type[int], type[integer]]] = ...

###

@overload
def fftshift(x: _ArrayLike[_ScalarT], axes: _ShapeLike | None = None) -> NDArray[_ScalarT]: ...
@overload
def fftshift(x: ArrayLike, axes: _ShapeLike | None = None) -> NDArray[Any]: ...

#
@overload
def ifftshift(x: _ArrayLike[_ScalarT], axes: _ShapeLike | None = None) -> NDArray[_ScalarT]: ...
@overload
def ifftshift(x: ArrayLike, axes: _ShapeLike | None = None) -> NDArray[Any]: ...

#
@overload
def fftfreq(n: int | integer, d: _ArrayLikeFloat_co = 1.0, device: L["cpu"] | None = None) -> NDArray[floating]: ...
@overload
def fftfreq(n: int | integer, d: _ArrayLikeComplex_co = 1.0, device: L["cpu"] | None = None) -> NDArray[complexfloating]: ...

#
@overload
def rfftfreq(n: int | integer, d: _ArrayLikeFloat_co = 1.0, device: L["cpu"] | None = None) -> NDArray[floating]: ...
@overload
def rfftfreq(n: int | integer, d: _ArrayLikeComplex_co = 1.0, device: L["cpu"] | None = None) -> NDArray[complexfloating]: ...
