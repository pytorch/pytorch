from collections.abc import Iterable
from typing import (
    Any,
    NamedTuple,
    Never,
    SupportsIndex,
    SupportsInt,
    TypeAlias,
    TypeVar,
    overload,
)
from typing import Literal as L

import numpy as np
from numpy import (
    complex128,
    complexfloating,
    float64,
    # other
    floating,
    int32,
    object_,
    signedinteger,
    timedelta64,
    unsignedinteger,
    # re-exports
    vecdot,
)
from numpy._core.fromnumeric import matrix_transpose
from numpy._core.numeric import tensordot
from numpy._globals import _NoValueType
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
    _ArrayLikeTD64_co,
    _ArrayLikeUInt_co,
)
from numpy.linalg import LinAlgError

__all__ = [
    "matrix_power",
    "solve",
    "tensorsolve",
    "tensorinv",
    "inv",
    "cholesky",
    "eigvals",
    "eigvalsh",
    "pinv",
    "slogdet",
    "det",
    "svd",
    "svdvals",
    "eig",
    "eigh",
    "lstsq",
    "norm",
    "qr",
    "cond",
    "matrix_rank",
    "LinAlgError",
    "multi_dot",
    "trace",
    "diagonal",
    "cross",
    "outer",
    "tensordot",
    "matmul",
    "matrix_transpose",
    "matrix_norm",
    "vector_norm",
    "vecdot",
]

_NumberT = TypeVar("_NumberT", bound=np.number)

_ModeKind: TypeAlias = L["reduced", "complete", "r", "raw"]

###

fortran_int = np.intc

class EigResult(NamedTuple):
    eigenvalues: NDArray[Any]
    eigenvectors: NDArray[Any]

class EighResult(NamedTuple):
    eigenvalues: NDArray[Any]
    eigenvectors: NDArray[Any]

class QRResult(NamedTuple):
    Q: NDArray[Any]
    R: NDArray[Any]

class SlogdetResult(NamedTuple):
    # TODO: `sign` and `logabsdet` are scalars for input 2D arrays and
    # a `(x.ndim - 2)`` dimensionl arrays otherwise
    sign: Any
    logabsdet: Any

class SVDResult(NamedTuple):
    U: NDArray[Any]
    S: NDArray[Any]
    Vh: NDArray[Any]

@overload
def tensorsolve(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
    axes: Iterable[int] | None = ...,
) -> NDArray[float64]: ...
@overload
def tensorsolve(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    axes: Iterable[int] | None = ...,
) -> NDArray[floating]: ...
@overload
def tensorsolve(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
    axes: Iterable[int] | None = ...,
) -> NDArray[complexfloating]: ...

@overload
def solve(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
) -> NDArray[float64]: ...
@overload
def solve(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
) -> NDArray[floating]: ...
@overload
def solve(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
) -> NDArray[complexfloating]: ...

@overload
def tensorinv(
    a: _ArrayLikeInt_co,
    ind: int = ...,
) -> NDArray[float64]: ...
@overload
def tensorinv(
    a: _ArrayLikeFloat_co,
    ind: int = ...,
) -> NDArray[floating]: ...
@overload
def tensorinv(
    a: _ArrayLikeComplex_co,
    ind: int = ...,
) -> NDArray[complexfloating]: ...

@overload
def inv(a: _ArrayLikeInt_co) -> NDArray[float64]: ...
@overload
def inv(a: _ArrayLikeFloat_co) -> NDArray[floating]: ...
@overload
def inv(a: _ArrayLikeComplex_co) -> NDArray[complexfloating]: ...

# TODO: The supported input and output dtypes are dependent on the value of `n`.
# For example: `n < 0` always casts integer types to float64
def matrix_power(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    n: SupportsIndex,
) -> NDArray[Any]: ...

@overload
def cholesky(a: _ArrayLikeInt_co, /, *, upper: bool = False) -> NDArray[float64]: ...
@overload
def cholesky(a: _ArrayLikeFloat_co, /, *, upper: bool = False) -> NDArray[floating]: ...
@overload
def cholesky(a: _ArrayLikeComplex_co, /, *, upper: bool = False) -> NDArray[complexfloating]: ...

@overload
def outer(x1: _ArrayLike[Never], x2: _ArrayLike[Never], /) -> NDArray[Any]: ...
@overload
def outer(x1: _ArrayLikeBool_co, x2: _ArrayLikeBool_co, /) -> NDArray[np.bool]: ...
@overload
def outer(x1: _ArrayLike[_NumberT], x2: _ArrayLike[_NumberT], /) -> NDArray[_NumberT]: ...
@overload
def outer(x1: _ArrayLikeUInt_co, x2: _ArrayLikeUInt_co, /) -> NDArray[unsignedinteger]: ...
@overload
def outer(x1: _ArrayLikeInt_co, x2: _ArrayLikeInt_co, /) -> NDArray[signedinteger]: ...
@overload
def outer(x1: _ArrayLikeFloat_co, x2: _ArrayLikeFloat_co, /) -> NDArray[floating]: ...
@overload
def outer(x1: _ArrayLikeComplex_co, x2: _ArrayLikeComplex_co, /) -> NDArray[complexfloating]: ...
@overload
def outer(x1: _ArrayLikeTD64_co, x2: _ArrayLikeTD64_co, /) -> NDArray[timedelta64]: ...
@overload
def outer(x1: _ArrayLikeObject_co, x2: _ArrayLikeObject_co, /) -> NDArray[object_]: ...
@overload
def outer(
    x1: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    x2: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    /,
) -> NDArray[Any]: ...

@overload
def qr(a: _ArrayLikeInt_co, mode: _ModeKind = ...) -> QRResult: ...
@overload
def qr(a: _ArrayLikeFloat_co, mode: _ModeKind = ...) -> QRResult: ...
@overload
def qr(a: _ArrayLikeComplex_co, mode: _ModeKind = ...) -> QRResult: ...

@overload
def eigvals(a: _ArrayLikeInt_co) -> NDArray[float64] | NDArray[complex128]: ...
@overload
def eigvals(a: _ArrayLikeFloat_co) -> NDArray[floating] | NDArray[complexfloating]: ...
@overload
def eigvals(a: _ArrayLikeComplex_co) -> NDArray[complexfloating]: ...

@overload
def eigvalsh(a: _ArrayLikeInt_co, UPLO: L["L", "U", "l", "u"] = ...) -> NDArray[float64]: ...
@overload
def eigvalsh(a: _ArrayLikeComplex_co, UPLO: L["L", "U", "l", "u"] = ...) -> NDArray[floating]: ...

@overload
def eig(a: _ArrayLikeInt_co) -> EigResult: ...
@overload
def eig(a: _ArrayLikeFloat_co) -> EigResult: ...
@overload
def eig(a: _ArrayLikeComplex_co) -> EigResult: ...

@overload
def eigh(
    a: _ArrayLikeInt_co,
    UPLO: L["L", "U", "l", "u"] = ...,
) -> EighResult: ...
@overload
def eigh(
    a: _ArrayLikeFloat_co,
    UPLO: L["L", "U", "l", "u"] = ...,
) -> EighResult: ...
@overload
def eigh(
    a: _ArrayLikeComplex_co,
    UPLO: L["L", "U", "l", "u"] = ...,
) -> EighResult: ...

@overload
def svd(
    a: _ArrayLikeInt_co,
    full_matrices: bool = ...,
    compute_uv: L[True] = ...,
    hermitian: bool = ...,
) -> SVDResult: ...
@overload
def svd(
    a: _ArrayLikeFloat_co,
    full_matrices: bool = ...,
    compute_uv: L[True] = ...,
    hermitian: bool = ...,
) -> SVDResult: ...
@overload
def svd(
    a: _ArrayLikeComplex_co,
    full_matrices: bool = ...,
    compute_uv: L[True] = ...,
    hermitian: bool = ...,
) -> SVDResult: ...
@overload
def svd(
    a: _ArrayLikeInt_co,
    full_matrices: bool = ...,
    compute_uv: L[False] = ...,
    hermitian: bool = ...,
) -> NDArray[float64]: ...
@overload
def svd(
    a: _ArrayLikeComplex_co,
    full_matrices: bool = ...,
    compute_uv: L[False] = ...,
    hermitian: bool = ...,
) -> NDArray[floating]: ...

def svdvals(
    x: _ArrayLikeInt_co | _ArrayLikeFloat_co | _ArrayLikeComplex_co
) -> NDArray[floating]: ...

# TODO: Returns a scalar for 2D arrays and
# a `(x.ndim - 2)`` dimensionl array otherwise
def cond(x: _ArrayLikeComplex_co, p: float | L["fro", "nuc"] | None = ...) -> Any: ...

# TODO: Returns `int` for <2D arrays and `intp` otherwise
def matrix_rank(
    A: _ArrayLikeComplex_co,
    tol: _ArrayLikeFloat_co | None = ...,
    hermitian: bool = ...,
    *,
    rtol: _ArrayLikeFloat_co | None = ...,
) -> Any: ...

@overload
def pinv(
    a: _ArrayLikeInt_co,
    rcond: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | _NoValueType = ...,
) -> NDArray[float64]: ...
@overload
def pinv(
    a: _ArrayLikeFloat_co,
    rcond: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | _NoValueType = ...,
) -> NDArray[floating]: ...
@overload
def pinv(
    a: _ArrayLikeComplex_co,
    rcond: _ArrayLikeFloat_co | None = None,
    hermitian: bool = False,
    *,
    rtol: _ArrayLikeFloat_co | _NoValueType = ...,
) -> NDArray[complexfloating]: ...

# TODO: Returns a 2-tuple of scalars for 2D arrays and
# a 2-tuple of `(a.ndim - 2)`` dimensionl arrays otherwise
def slogdet(a: _ArrayLikeComplex_co) -> SlogdetResult: ...

# TODO: Returns a 2-tuple of scalars for 2D arrays and
# a 2-tuple of `(a.ndim - 2)`` dimensionl arrays otherwise
def det(a: _ArrayLikeComplex_co) -> Any: ...

@overload
def lstsq(a: _ArrayLikeInt_co, b: _ArrayLikeInt_co, rcond: float | None = ...) -> tuple[
    NDArray[float64],
    NDArray[float64],
    int32,
    NDArray[float64],
]: ...
@overload
def lstsq(a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, rcond: float | None = ...) -> tuple[
    NDArray[floating],
    NDArray[floating],
    int32,
    NDArray[floating],
]: ...
@overload
def lstsq(a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co, rcond: float | None = ...) -> tuple[
    NDArray[complexfloating],
    NDArray[floating],
    int32,
    NDArray[floating],
]: ...

@overload
def norm(
    x: ArrayLike,
    ord: float | L["fro", "nuc"] | None = ...,
    axis: None = ...,
    keepdims: bool = ...,
) -> floating: ...
@overload
def norm(
    x: ArrayLike,
    ord: float | L["fro", "nuc"] | None = ...,
    axis: SupportsInt | SupportsIndex | tuple[int, ...] = ...,
    keepdims: bool = ...,
) -> Any: ...

@overload
def matrix_norm(
    x: ArrayLike,
    /,
    *,
    ord: float | L["fro", "nuc"] | None = ...,
    keepdims: bool = ...,
) -> floating: ...
@overload
def matrix_norm(
    x: ArrayLike,
    /,
    *,
    ord: float | L["fro", "nuc"] | None = ...,
    keepdims: bool = ...,
) -> Any: ...

@overload
def vector_norm(
    x: ArrayLike,
    /,
    *,
    axis: None = ...,
    ord: float | None = ...,
    keepdims: bool = ...,
) -> floating: ...
@overload
def vector_norm(
    x: ArrayLike,
    /,
    *,
    axis: SupportsInt | SupportsIndex | tuple[int, ...] = ...,
    ord: float | None = ...,
    keepdims: bool = ...,
) -> Any: ...

# TODO: Returns a scalar or array
def multi_dot(
    arrays: Iterable[_ArrayLikeComplex_co | _ArrayLikeObject_co | _ArrayLikeTD64_co],
    *,
    out: NDArray[Any] | None = ...,
) -> Any: ...

def diagonal(
    x: ArrayLike,  # >= 2D array
    /,
    *,
    offset: SupportsIndex = ...,
) -> NDArray[Any]: ...

def trace(
    x: ArrayLike,  # >= 2D array
    /,
    *,
    offset: SupportsIndex = ...,
    dtype: DTypeLike = ...,
) -> Any: ...

@overload
def cross(
    x1: _ArrayLikeUInt_co,
    x2: _ArrayLikeUInt_co,
    /,
    *,
    axis: int = ...,
) -> NDArray[unsignedinteger]: ...
@overload
def cross(
    x1: _ArrayLikeInt_co,
    x2: _ArrayLikeInt_co,
    /,
    *,
    axis: int = ...,
) -> NDArray[signedinteger]: ...
@overload
def cross(
    x1: _ArrayLikeFloat_co,
    x2: _ArrayLikeFloat_co,
    /,
    *,
    axis: int = ...,
) -> NDArray[floating]: ...
@overload
def cross(
    x1: _ArrayLikeComplex_co,
    x2: _ArrayLikeComplex_co,
    /,
    *,
    axis: int = ...,
) -> NDArray[complexfloating]: ...

@overload
def matmul(x1: _ArrayLike[_NumberT], x2: _ArrayLike[_NumberT], /) -> NDArray[_NumberT]: ...
@overload
def matmul(x1: _ArrayLikeInt_co, x2: _ArrayLikeInt_co, /) -> NDArray[signedinteger]: ...
@overload
def matmul(x1: _ArrayLikeUInt_co, x2: _ArrayLikeUInt_co, /) -> NDArray[unsignedinteger]: ...
@overload
def matmul(x1: _ArrayLikeFloat_co, x2: _ArrayLikeFloat_co, /) -> NDArray[floating]: ...
@overload
def matmul(x1: _ArrayLikeComplex_co, x2: _ArrayLikeComplex_co, /) -> NDArray[complexfloating]: ...
