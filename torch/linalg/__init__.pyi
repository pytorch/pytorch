# Stub file for torch.linalg module
# Type annotations for PyTorch Linear Algebra functions

from collections.abc import Sequence
from typing import Literal, Optional, Union

import torch.return_types
from torch import SymInt, Tensor
from torch._C import dtype
from torch.types import _float, _int

# Exception class
class LinAlgError(RuntimeError): ...

# Common notes dictionary
common_notes: dict[str, str]

# Core linear algebra functions
def cross(
    input: Tensor, other: Tensor, *, dim: int = -1, out: Optional[Tensor] = None
) -> Tensor: ...
def cholesky(
    A: Tensor, *, upper: bool = False, out: Optional[Tensor] = None
) -> Tensor: ...
def cholesky_ex(
    A: Tensor,
    *,
    upper: bool = False,
    check_errors: bool = False,
    out: Optional[tuple[Tensor, Tensor]] = None,
) -> torch.return_types._lu_with_info: ...
def cond(
    A: Tensor,
    p: Optional[Union[_int, _float, str]] = None,
    *,
    out: Optional[Tensor] = None,
) -> Tensor: ...
def det(A: Tensor, *, out: Optional[Tensor] = None) -> Tensor: ...
def diagonal(
    A: Tensor, *, offset: int = 0, dim1: int = -2, dim2: int = -1
) -> Tensor: ...
def eig(
    A: Tensor, *, out: Optional[tuple[Tensor, Tensor]] = None
) -> tuple[Tensor, Tensor]: ...
def eigh(
    A: Tensor, UPLO: str = "L", *, out: Optional[tuple[Tensor, Tensor]] = None
) -> tuple[Tensor, Tensor]: ...
def eigvals(A: Tensor, *, out: Optional[Tensor] = None) -> Tensor: ...
def eigvalsh(A: Tensor, UPLO: str = "L", *, out: Optional[Tensor] = None) -> Tensor: ...
def householder_product(
    A: Tensor, tau: Tensor, *, out: Optional[Tensor] = None
) -> Tensor: ...
def inv(A: Tensor, *, out: Optional[Tensor] = None) -> Tensor: ...
def inv_ex(
    A: Tensor,
    *,
    check_errors: bool = False,
    out: Optional[tuple[Tensor, Tensor]] = None,
) -> tuple[Tensor, Tensor]: ...
def ldl_factor(
    A: Tensor, *, hermitian: bool = False, out: Optional[tuple[Tensor, Tensor]] = None
) -> tuple[Tensor, Tensor]: ...
def ldl_factor_ex(
    A: Tensor,
    *,
    hermitian: bool = False,
    check_errors: bool = False,
    out: Optional[tuple[Tensor, Tensor, Tensor]] = None,
) -> tuple[Tensor, Tensor, Tensor]: ...
def ldl_solve(
    LD: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    hermitian: bool = False,
    out: Optional[Tensor] = None,
) -> Tensor: ...
def lstsq(
    A: Tensor,
    B: Tensor,
    rcond: Optional[_float] = None,
    *,
    driver: Optional[str] = None,
    out: Optional[tuple[Tensor, Tensor, Tensor, Tensor]] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...
def lu(
    A: Tensor,
    *,
    pivot: bool = True,
    out: Optional[tuple[Tensor, Tensor, Tensor]] = None,
) -> tuple[Tensor, Tensor, Tensor]: ...
def lu_factor(
    A: Tensor, *, pivot: bool = True, out: Optional[tuple[Tensor, Tensor]] = None
) -> tuple[Tensor, Tensor]: ...
def lu_factor_ex(
    A: Tensor,
    *,
    pivot: bool = True,
    check_errors: bool = False,
    out: Optional[tuple[Tensor, Tensor, Tensor]] = None,
) -> tuple[Tensor, Tensor, Tensor]: ...
def lu_solve(
    LU: Tensor,
    pivots: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    adjoint: bool = False,
    out: Optional[Tensor] = None,
) -> Tensor: ...
def matmul(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor: ...
def matrix_exp(A: Tensor, *, out: Optional[Tensor] = None) -> Tensor: ...
def matrix_norm(
    A: Tensor,
    ord: Union[_int, _float, str] = "fro",
    dim: Union[int, tuple[int, int], Sequence[Union[int, SymInt]]] = (-2, -1),
    keepdim: bool = False,
    *,
    dtype: Optional[dtype] = None,
    out: Optional[Tensor] = None,
) -> Tensor: ...
def matrix_power(A: Tensor, n: int, *, out: Optional[Tensor] = None) -> Tensor: ...
def matrix_rank(
    A: Tensor,
    tol: Optional[_float] = None,
    hermitian: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> Tensor: ...
def multi_dot(tensors: list[Tensor], *, out: Optional[Tensor] = None) -> Tensor: ...
def norm(
    A: Tensor,
    ord: Optional[Union[_int, _float, str]] = None,
    dim: Optional[Union[int, Sequence[Union[int, SymInt]]]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[dtype] = None,
    out: Optional[Tensor] = None,
) -> Tensor: ...
def pinv(
    A: Tensor,
    rcond: Optional[_float] = None,
    hermitian: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> Tensor: ...
def qr(
    A: Tensor,
    mode: Literal["reduced", "complete", "r"] = "reduced",
    *,
    out: Optional[tuple[Tensor, Tensor]] = None,
) -> torch.return_types.qr: ...
def slogdet(
    A: Tensor, *, out: Optional[tuple[Tensor, Tensor]] = None
) -> torch.return_types.slogdet: ...
def solve(
    A: Tensor, B: Tensor, *, left: bool = True, out: Optional[Tensor] = None
) -> Tensor: ...
def solve_ex(
    A: Tensor,
    B: Tensor,
    *,
    left: bool = True,
    check_errors: bool = False,
    out: Optional[tuple[Tensor, Tensor]] = None,
) -> tuple[Tensor, Tensor]: ...
def solve_triangular(
    A: Tensor,
    B: Tensor,
    *,
    upper: bool,
    left: bool = True,
    unitriangular: bool = False,
    out: Optional[Tensor] = None,
) -> Tensor: ...
def svd(
    A: Tensor,
    full_matrices: bool = True,
    *,
    driver: Optional[str] = None,
    out: Optional[tuple[Tensor, Tensor, Tensor]] = None,
) -> tuple[Tensor, Tensor, Tensor]: ...
def svdvals(
    A: Tensor, *, driver: Optional[str] = None, out: Optional[Tensor] = None
) -> Tensor: ...
def tensorinv(A: Tensor, ind: int = 2, *, out: Optional[Tensor] = None) -> Tensor: ...
def tensorsolve(
    A: Tensor,
    B: Tensor,
    dims: Optional[Sequence[int]] = None,
    *,
    out: Optional[Tensor] = None,
) -> Tensor: ...
def vander(
    x: Tensor, N: Optional[int] = None, *, out: Optional[Tensor] = None
) -> Tensor: ...
def vecdot(
    x: Tensor, y: Tensor, *, dim: int = -1, out: Optional[Tensor] = None
) -> Tensor: ...
def vector_norm(
    x: Tensor,
    ord: Optional[Union[_int, _float, complex]] = 2,
    dim: Optional[Union[int, Sequence[Union[int, SymInt]]]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[dtype] = None,
    out: Optional[Tensor] = None,
) -> Tensor: ...
