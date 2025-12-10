from collections.abc import Callable, Sequence
from typing import (
    Any,
    Concatenate,
    ParamSpec,
    Protocol,
    SupportsIndex,
    TypeVar,
    overload,
    type_check_only,
)

from typing_extensions import deprecated

import numpy as np
from numpy import (
    _CastingKind,
    complexfloating,
    floating,
    generic,
    integer,
    object_,
    signedinteger,
    ufunc,
    unsignedinteger,
)
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _ArrayLikeUInt_co,
    _ShapeLike,
)

__all__ = [
    "column_stack",
    "row_stack",
    "dstack",
    "array_split",
    "split",
    "hsplit",
    "vsplit",
    "dsplit",
    "apply_over_axes",
    "expand_dims",
    "apply_along_axis",
    "kron",
    "tile",
    "take_along_axis",
    "put_along_axis",
]

_P = ParamSpec("_P")
_ScalarT = TypeVar("_ScalarT", bound=generic)

# Signature of `__array_wrap__`
@type_check_only
class _ArrayWrap(Protocol):
    def __call__(
        self,
        array: NDArray[Any],
        context: tuple[ufunc, tuple[Any, ...], int] | None = ...,
        return_scalar: bool = ...,
        /,
    ) -> Any: ...

@type_check_only
class _SupportsArrayWrap(Protocol):
    @property
    def __array_wrap__(self) -> _ArrayWrap: ...

###

def take_along_axis(
    arr: _ScalarT | NDArray[_ScalarT],
    indices: NDArray[integer],
    axis: int | None = ...,
) -> NDArray[_ScalarT]: ...

def put_along_axis(
    arr: NDArray[_ScalarT],
    indices: NDArray[integer],
    values: ArrayLike,
    axis: int | None,
) -> None: ...

@overload
def apply_along_axis(
    func1d: Callable[Concatenate[NDArray[Any], _P], _ArrayLike[_ScalarT]],
    axis: SupportsIndex,
    arr: ArrayLike,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> NDArray[_ScalarT]: ...
@overload
def apply_along_axis(
    func1d: Callable[Concatenate[NDArray[Any], _P], Any],
    axis: SupportsIndex,
    arr: ArrayLike,
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> NDArray[Any]: ...

def apply_over_axes(
    func: Callable[[NDArray[Any], int], NDArray[_ScalarT]],
    a: ArrayLike,
    axes: int | Sequence[int],
) -> NDArray[_ScalarT]: ...

@overload
def expand_dims(
    a: _ArrayLike[_ScalarT],
    axis: _ShapeLike,
) -> NDArray[_ScalarT]: ...
@overload
def expand_dims(
    a: ArrayLike,
    axis: _ShapeLike,
) -> NDArray[Any]: ...

# Deprecated in NumPy 2.0, 2023-08-18
@deprecated("`row_stack` alias is deprecated. Use `np.vstack` directly.")
def row_stack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: _CastingKind = "same_kind",
) -> NDArray[Any]: ...

#
@overload
def column_stack(tup: Sequence[_ArrayLike[_ScalarT]]) -> NDArray[_ScalarT]: ...
@overload
def column_stack(tup: Sequence[ArrayLike]) -> NDArray[Any]: ...

@overload
def dstack(tup: Sequence[_ArrayLike[_ScalarT]]) -> NDArray[_ScalarT]: ...
@overload
def dstack(tup: Sequence[ArrayLike]) -> NDArray[Any]: ...

@overload
def array_split(
    ary: _ArrayLike[_ScalarT],
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = ...,
) -> list[NDArray[_ScalarT]]: ...
@overload
def array_split(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = ...,
) -> list[NDArray[Any]]: ...

@overload
def split(
    ary: _ArrayLike[_ScalarT],
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = ...,
) -> list[NDArray[_ScalarT]]: ...
@overload
def split(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = ...,
) -> list[NDArray[Any]]: ...

@overload
def hsplit(
    ary: _ArrayLike[_ScalarT],
    indices_or_sections: _ShapeLike,
) -> list[NDArray[_ScalarT]]: ...
@overload
def hsplit(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
) -> list[NDArray[Any]]: ...

@overload
def vsplit(
    ary: _ArrayLike[_ScalarT],
    indices_or_sections: _ShapeLike,
) -> list[NDArray[_ScalarT]]: ...
@overload
def vsplit(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
) -> list[NDArray[Any]]: ...

@overload
def dsplit(
    ary: _ArrayLike[_ScalarT],
    indices_or_sections: _ShapeLike,
) -> list[NDArray[_ScalarT]]: ...
@overload
def dsplit(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
) -> list[NDArray[Any]]: ...

@overload
def get_array_wrap(*args: _SupportsArrayWrap) -> _ArrayWrap: ...
@overload
def get_array_wrap(*args: object) -> _ArrayWrap | None: ...

@overload
def kron(a: _ArrayLikeBool_co, b: _ArrayLikeBool_co) -> NDArray[np.bool]: ...  # type: ignore[misc]
@overload
def kron(a: _ArrayLikeUInt_co, b: _ArrayLikeUInt_co) -> NDArray[unsignedinteger]: ...  # type: ignore[misc]
@overload
def kron(a: _ArrayLikeInt_co, b: _ArrayLikeInt_co) -> NDArray[signedinteger]: ...  # type: ignore[misc]
@overload
def kron(a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co) -> NDArray[floating]: ...  # type: ignore[misc]
@overload
def kron(a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co) -> NDArray[complexfloating]: ...
@overload
def kron(a: _ArrayLikeObject_co, b: Any) -> NDArray[object_]: ...
@overload
def kron(a: Any, b: _ArrayLikeObject_co) -> NDArray[object_]: ...

@overload
def tile(
    A: _ArrayLike[_ScalarT],
    reps: int | Sequence[int],
) -> NDArray[_ScalarT]: ...
@overload
def tile(
    A: ArrayLike,
    reps: int | Sequence[int],
) -> NDArray[Any]: ...
