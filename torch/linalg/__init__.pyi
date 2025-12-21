# Stub file for torch.linalg
from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import SymInt, Tensor

# Return type classes
class SVDResult(NamedTuple):
    U: Tensor
    S: Tensor
    Vh: Tensor

class EigResult(NamedTuple):
    eigenvalues: Tensor
    eigenvectors: Tensor

class EighResult(NamedTuple):
    eigenvalues: Tensor
    eigenvectors: Tensor

class QRResult(NamedTuple):
    Q: Tensor
    R: Tensor

class LUResult(NamedTuple):
    LU: Tensor
    pivots: Tensor

class SlogdetResult(NamedTuple):
    sign: Tensor
    logabsdet: Tensor

class LstsqResult(NamedTuple):
    solution: Tensor
    residuals: Tensor
    rank: Tensor
    singular_values: Tensor

class CholeskyResult(NamedTuple):
    L: Tensor
    info: Tensor

# Norms
def norm(
    input: Tensor,
    ord: float | int | str | None = None,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
    out: Tensor | None = None,
) -> Tensor: ...
def vector_norm(
    input: Tensor,
    ord: float | int | complex | bool | None = 2,
    dim: int | SymInt | Sequence[int | SymInt] | None = None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
    out: Tensor | None = None,
) -> Tensor: ...
def matrix_norm(
    input: Tensor,
    ord: float | int | str = "fro",
    dim: tuple[int, int] | Sequence[int | SymInt] = (-2, -1),
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
    out: Tensor | None = None,
) -> Tensor: ...

# Decompositions
def svd(
    input: Tensor,
    full_matrices: bool = True,
    *,
    driver: str | None = None,
    out: tuple[Tensor, Tensor, Tensor] | None = None,
) -> SVDResult: ...
def svdvals(
    input: Tensor,
    *,
    driver: str | None = None,
    out: Tensor | None = None,
) -> Tensor: ...
def eig(input: Tensor, *, out: tuple[Tensor, Tensor] | None = None) -> EigResult: ...
def eigvals(input: Tensor, *, out: Tensor | None = None) -> Tensor: ...
def eigh(
    input: Tensor,
    UPLO: str = "L",
    *,
    out: tuple[Tensor, Tensor] | None = None,
) -> EighResult: ...
def eigvalsh(
    input: Tensor,
    UPLO: str = "L",
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def qr(
    input: Tensor,
    mode: str = "reduced",
    *,
    out: tuple[Tensor, Tensor] | None = None,
) -> QRResult: ...
def lu(
    input: Tensor,
    *,
    pivot: bool = True,
    out: tuple[Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor]: ...
def lu_factor(
    input: Tensor,
    *,
    pivot: bool = True,
    out: tuple[Tensor, Tensor] | None = None,
) -> LUResult: ...
def cholesky(
    input: Tensor,
    *,
    upper: bool = False,
    out: Tensor | None = None,
) -> Tensor: ...
def cholesky_ex(
    input: Tensor,
    *,
    upper: bool = False,
    check_errors: bool = False,
    out: tuple[Tensor, Tensor] | None = None,
) -> CholeskyResult: ...

# Matrix properties
def det(input: Tensor, *, out: Tensor | None = None) -> Tensor: ...
def slogdet(
    input: Tensor, *, out: tuple[Tensor, Tensor] | None = None
) -> SlogdetResult: ...
def logdet(input: Tensor) -> Tensor: ...
def matrix_rank(
    input: Tensor,
    *,
    atol: float | Tensor | None = None,
    rtol: float | Tensor | None = None,
    hermitian: bool = False,
    out: Tensor | None = None,
) -> Tensor: ...
def cond(
    input: Tensor,
    p: float | int | str | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def matrix_power(input: Tensor, n: int, *, out: Tensor | None = None) -> Tensor: ...
def matrix_exp(input: Tensor) -> Tensor: ...

# Inverses and solvers
def inv(input: Tensor, *, out: Tensor | None = None) -> Tensor: ...
def inv_ex(
    input: Tensor,
    *,
    check_errors: bool = False,
    out: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor]: ...
def pinv(
    input: Tensor,
    *,
    atol: float | Tensor | None = None,
    rtol: float | Tensor | None = None,
    hermitian: bool = False,
    out: Tensor | None = None,
) -> Tensor: ...
def solve(
    input: Tensor,
    other: Tensor,
    *,
    left: bool = True,
    out: Tensor | None = None,
) -> Tensor: ...
def solve_ex(
    input: Tensor,
    other: Tensor,
    *,
    left: bool = True,
    check_errors: bool = False,
    out: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor]: ...
def lstsq(
    input: Tensor,
    b: Tensor,
    rcond: float | None = None,
    *,
    driver: str | None = None,
) -> LstsqResult: ...
def solve_triangular(
    input: Tensor,
    B: Tensor,
    *,
    upper: bool,
    left: bool = True,
    unitriangular: bool = False,
    out: Tensor | None = None,
) -> Tensor: ...
def lu_solve(
    LU: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    adjoint: bool = False,
    out: Tensor | None = None,
) -> Tensor: ...
def cholesky_solve(
    input: Tensor,
    input2: Tensor,
    *,
    upper: bool = False,
    out: Tensor | None = None,
) -> Tensor: ...

# Products
def matmul(input: Tensor, other: Tensor, *, out: Tensor | None = None) -> Tensor: ...
def multi_dot(tensors: list[Tensor], *, out: Tensor | None = None) -> Tensor: ...
def cross(
    input: Tensor,
    other: Tensor,
    *,
    dim: int = -1,
    out: Tensor | None = None,
) -> Tensor: ...
def householder_product(
    input: Tensor,
    tau: Tensor,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def tensordot(
    a: Tensor,
    b: Tensor,
    dims: int | tuple[list[int], list[int]],
    out: Tensor | None = None,
) -> Tensor: ...
def tensorinv(input: Tensor, ind: int = 2, *, out: Tensor | None = None) -> Tensor: ...
def tensorsolve(
    input: Tensor,
    other: Tensor,
    dims: tuple[int, ...] | None = None,
    *,
    out: Tensor | None = None,
) -> Tensor: ...
def vander(x: Tensor, N: int | None = None, *, increasing: bool = False) -> Tensor: ...

# Misc
def diagonal(
    input: Tensor,
    *,
    offset: int = 0,
    dim1: int = -2,
    dim2: int = -1,
) -> Tensor: ...
def trace(input: Tensor) -> Tensor: ...
def vecdot(
    x: Tensor,
    y: Tensor,
    *,
    dim: int = -1,
    out: Tensor | None = None,
) -> Tensor: ...
