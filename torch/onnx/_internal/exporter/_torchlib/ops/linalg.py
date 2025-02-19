"""torch.ops.aten operators under the `linalg` module."""
# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value,type-var,operator,no-untyped-def,index"
# ruff: noqa: TCH001,TCH002
# flake8: noqa

from __future__ import annotations

import math
from typing import Optional, Sequence

from onnxscript import BOOL
from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import TensorType

import torch
from torch.onnx._internal.exporter._torchlib._tensor_typing import TFloat, TTensor
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


aten = torch.ops.aten


def aten_linalg_cholesky(self: TensorType, upper: bool = False) -> TensorType:
    """linalg_cholesky(Tensor self, *, bool upper=False) -> Tensor"""

    raise NotImplementedError


def aten_linalg_cholesky_ex(
    self: TensorType, upper: bool = False, check_errors: bool = False
) -> tuple[TensorType, TensorType]:
    """linalg_cholesky_ex(Tensor self, *, bool upper=False, bool check_errors=False) -> (Tensor L, Tensor info)"""

    raise NotImplementedError


def aten_linalg_cond(self: TensorType, p: Optional[float] = None) -> TensorType:
    """linalg_cond(Tensor self, Scalar? p=None) -> Tensor"""

    raise NotImplementedError


def aten_linalg_cross(self: TTensor, other: TTensor, dim: int = -1) -> TTensor:
    """linalg_cross(Tensor self, Tensor other, *, int dim=-1) -> Tensor"""

    # Same implementation as aten_cross
    raise NotImplementedError


@onnx_impl((aten._linalg_det.default, aten.linalg_det.default, aten.det.default))
def aten_linalg_det(A: TFloat) -> TFloat:
    """linalg_det(Tensor A) -> Tensor"""

    return op.Det(A)


def aten_linalg_diagonal(
    A: TensorType, offset: int = 0, dim1: int = -2, dim2: int = -1
) -> TensorType:
    """linalg_diagonal(Tensor(a) A, *, int offset=0, int dim1=-2, int dim2=-1) -> Tensor(a)"""

    raise NotImplementedError


def aten_linalg_eig(self: TensorType) -> tuple[TensorType, TensorType]:
    """linalg_eig(Tensor self) -> (Tensor eigenvalues, Tensor eigenvectors)"""

    raise NotImplementedError


def aten_linalg_eigh(
    self: TensorType, UPLO: str = "L"
) -> tuple[TensorType, TensorType]:
    """linalg_eigh(Tensor self, str UPLO="L") -> (Tensor eigenvalues, Tensor eigenvectors)"""

    raise NotImplementedError


def aten_linalg_eigvals(self: TensorType) -> TensorType:
    """linalg_eigvals(Tensor self) -> Tensor"""

    raise NotImplementedError


def aten_linalg_eigvalsh(self: TensorType, UPLO: str = "L") -> TensorType:
    """linalg_eigvalsh(Tensor self, str UPLO="L") -> Tensor"""

    raise NotImplementedError


def aten_linalg_householder_product(input: TensorType, tau: TensorType) -> TensorType:
    """linalg_householder_product(Tensor input, Tensor tau) -> Tensor"""

    raise NotImplementedError


def aten_linalg_inv(A: TensorType) -> TensorType:
    """linalg_inv(Tensor A) -> Tensor"""

    raise NotImplementedError


def aten_linalg_inv_ex(
    A: TensorType, check_errors: bool = False
) -> tuple[TensorType, TensorType]:
    """linalg_inv_ex(Tensor A, *, bool check_errors=False) -> (Tensor inverse, Tensor info)"""

    raise NotImplementedError


def aten_linalg_ldl_factor(
    self: TensorType, hermitian: bool = False
) -> tuple[TensorType, TensorType]:
    """linalg_ldl_factor(Tensor self, *, bool hermitian=False) -> (Tensor LD, Tensor pivots)"""

    raise NotImplementedError


def aten_linalg_ldl_factor_ex(
    self: TensorType, hermitian: bool = False, check_errors: bool = False
) -> tuple[TensorType, TensorType, TensorType]:
    """linalg_ldl_factor_ex(Tensor self, *, bool hermitian=False, bool check_errors=False) -> (Tensor LD, Tensor pivots, Tensor info)"""

    raise NotImplementedError


def aten_linalg_ldl_solve(
    LD: TensorType, pivots: TensorType, B: TensorType, hermitian: bool = False
) -> TensorType:
    """linalg_ldl_solve(Tensor LD, Tensor pivots, Tensor B, *, bool hermitian=False) -> Tensor"""

    raise NotImplementedError


def aten_linalg_lstsq(
    self: TensorType,
    b: TensorType,
    rcond: Optional[float] = None,
    driver: Optional[str] = None,
) -> tuple[TensorType, TensorType, TensorType, TensorType]:
    """linalg_lstsq(Tensor self, Tensor b, float? rcond=None, *, str? driver=None) -> (Tensor solution, Tensor residuals, Tensor rank, Tensor singular_values)"""

    raise NotImplementedError


def aten_linalg_lu(
    A: TensorType, pivot: bool = True
) -> tuple[TensorType, TensorType, TensorType]:
    """linalg_lu(Tensor A, *, bool pivot=True) -> (Tensor P, Tensor L, Tensor U)"""

    raise NotImplementedError


def aten_linalg_lu_factor(
    A: TensorType, pivot: bool = True
) -> tuple[TensorType, TensorType]:
    """linalg_lu_factor(Tensor A, *, bool pivot=True) -> (Tensor LU, Tensor pivots)"""

    raise NotImplementedError


def aten_linalg_lu_factor_ex(
    A: TensorType, pivot: bool = True, check_errors: bool = False
) -> tuple[TensorType, TensorType, TensorType]:
    """linalg_lu_factor_ex(Tensor A, *, bool pivot=True, bool check_errors=False) -> (Tensor LU, Tensor pivots, Tensor info)"""

    raise NotImplementedError


def aten_linalg_lu_solve(
    LU: TensorType,
    pivots: TensorType,
    B: TensorType,
    left: bool = True,
    adjoint: bool = False,
) -> TensorType:
    """linalg_lu_solve(Tensor LU, Tensor pivots, Tensor B, *, bool left=True, bool adjoint=False) -> Tensor"""

    raise NotImplementedError


def aten_linalg_matmul(self: TensorType, other: TensorType) -> TensorType:
    """linalg_matmul(Tensor self, Tensor other) -> Tensor"""

    raise NotImplementedError


def aten_linalg_matrix_exp(self: TensorType) -> TensorType:
    """linalg_matrix_exp(Tensor self) -> Tensor"""

    raise NotImplementedError


def aten_linalg_matrix_norm(
    self: TensorType,
    ord: float,
    dim: Sequence[int] = (-2, -1),
    keepdim: bool = False,
    dtype: Optional[int] = None,
) -> TensorType:
    """linalg_matrix_norm(Tensor self, Scalar ord, int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""

    raise NotImplementedError


def aten_linalg_matrix_power(self: TensorType, n: int) -> TensorType:
    """linalg_matrix_power(Tensor self, int n) -> Tensor"""

    raise NotImplementedError


def aten_linalg_matrix_rank(
    self: TensorType, tol: float, hermitian: bool = False
) -> TensorType:
    """linalg_matrix_rank(Tensor self, float tol, bool hermitian=False) -> Tensor"""

    raise NotImplementedError


def aten_linalg_multi_dot(tensors: Sequence[TensorType]) -> TensorType:
    """linalg_multi_dot(Tensor[] tensors) -> Tensor"""

    raise NotImplementedError


def aten_linalg_norm(
    self: TensorType,
    ord: Optional[float] = None,
    dim: Optional[int] = None,
    keepdim: bool = False,
    dtype: Optional[int] = None,
) -> TensorType:
    """linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""

    raise NotImplementedError


def aten_linalg_pinv(
    self: TensorType, rcond: float, hermitian: bool = False
) -> TensorType:
    """linalg_pinv(Tensor self, float rcond, bool hermitian=False) -> Tensor"""

    raise NotImplementedError


def aten_linalg_qr(
    A: TensorType, mode: str = "reduced"
) -> tuple[TensorType, TensorType]:
    """linalg_qr(Tensor A, str mode='reduced') -> (Tensor Q, Tensor R)"""

    raise NotImplementedError


def aten_linalg_slogdet(A: TensorType) -> tuple[TensorType, TensorType]:
    """linalg_slogdet(Tensor A) -> (Tensor sign, Tensor logabsdet)"""

    raise NotImplementedError


def aten_linalg_solve(A: TensorType, B: TensorType, left: bool = True) -> TensorType:
    """linalg_solve(Tensor A, Tensor B, *, bool left=True) -> Tensor"""

    raise NotImplementedError


def aten_linalg_solve_ex(
    A: TensorType, B: TensorType, left: bool = True, check_errors: bool = False
) -> tuple[TensorType, TensorType]:
    """linalg_solve_ex(Tensor A, Tensor B, *, bool left=True, bool check_errors=False) -> (Tensor result, Tensor info)"""

    raise NotImplementedError


def aten_linalg_solve_triangular(
    self: TensorType,
    B: TensorType,
    upper: bool,
    left: bool = True,
    unitriangular: bool = False,
) -> TensorType:
    """linalg_solve_triangular(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False) -> Tensor"""

    raise NotImplementedError


def aten_linalg_svd(
    A: TensorType, full_matrices: bool = True, driver: Optional[str] = None
) -> tuple[TensorType, TensorType, TensorType]:
    """linalg_svd(Tensor A, bool full_matrices=True, *, str? driver=None) -> (Tensor U, Tensor S, Tensor Vh)"""

    raise NotImplementedError


def aten_linalg_svdvals(A: TensorType, driver: Optional[str] = None) -> TensorType:
    """linalg_svdvals(Tensor A, *, str? driver=None) -> Tensor"""

    raise NotImplementedError


def aten_linalg_tensorinv(self: TensorType, ind: int = 2) -> TensorType:
    """linalg_tensorinv(Tensor self, int ind=2) -> Tensor"""

    raise NotImplementedError


def aten_linalg_tensorsolve(
    self: TensorType, other: TensorType, dims: Optional[int] = None
) -> TensorType:
    """linalg_tensorsolve(Tensor self, Tensor other, int[]? dims=None) -> Tensor"""

    raise NotImplementedError


def aten_linalg_vander(x: TensorType, N: Optional[int] = None) -> TensorType:
    """linalg_vander(Tensor x, *, int? N=None) -> Tensor"""

    raise NotImplementedError


def aten_linalg_vecdot(x: TensorType, y: TensorType, dim: int = -1) -> TensorType:
    """linalg_vecdot(Tensor x, Tensor y, *, int dim=-1) -> Tensor"""

    raise NotImplementedError


@onnx_impl(aten.linalg_vector_norm.default, trace_only=True)
def aten_linalg_vector_norm(
    self: TFloat,
    ord: float = 2.0,
    dim: Optional[int] = None,
    keepdim: bool = False,
    dtype: int = -1,
) -> TFloat:
    """linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""

    if dtype != -1:
        self = op.Cast(self, to=dtype)
    if dim is None:
        self = op.Reshape(self, op.Constant(value_ints=[-1]))
        keepdim = False
    else:
        dim = op.Reshape(dim, op.Constant(value_ints=[-1]))
    self = op.Abs(self)
    if math.isinf(ord):
        if ord > 0:
            return op.ReduceMax(self, dim, keepdims=keepdim)
        else:
            return op.ReduceMin(self, dim, keepdims=keepdim)
    elif ord == 0.0:  # sum(x!=0) means count non-zero elements
        self_bool = op.Cast(self, to=BOOL.dtype)
        self_0_1 = op.CastLike(self_bool, self)
        return op.ReduceSum(self_0_1, dim, keepdims=keepdim)
    elif ord == 1.0:
        return op.ReduceL1(self, dim, keepdims=keepdim)
    elif ord == 2.0:
        return op.ReduceL2(self, dim, keepdims=keepdim)
    else:
        self_pow = op.Pow(self, ord)
        exp = op.CastLike(1 / ord, self)
        return op.Pow(op.ReduceSum(self_pow, dim, keepdims=keepdim), exp)
