# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR linalg translations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.onnx._internal.exporter._ir_builder import TranslationContext


aten = torch.ops.aten


@onnx_impl(aten.mm.default)
def aten_mm(op: TranslationContext, self: ir.Value, mat2: ir.Value) -> ir.Value:
    """mm(Tensor self, Tensor mat2) -> Tensor"""

    return op.MatMul(self, mat2)


@onnx_impl(aten.bmm.default)
def aten_bmm(op: TranslationContext, self: ir.Value, mat2: ir.Value) -> ir.Value:
    """bmm(Tensor self, Tensor mat2) -> Tensor"""

    return op.MatMul(self, mat2)


@onnx_impl(aten.mv.default)
def aten_mv(op: TranslationContext, self: ir.Value, vec: ir.Value) -> ir.Value:
    """mv(Tensor self, Tensor vec) -> Tensor"""

    return op.MatMul(self, vec)


@onnx_impl(aten.dot.default)
def aten_dot(op: TranslationContext, self: ir.Value, tensor: ir.Value) -> ir.Value:
    """dot(Tensor self, Tensor tensor) -> Tensor"""

    return op.MatMul(self, tensor)


@onnx_impl(aten.matmul.default)
def aten_matmul(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """matmul(Tensor self, Tensor other) -> Tensor"""

    return op.MatMul(self, other)


@onnx_impl(aten.matmul.default, complex=True)
def aten_matmul_complex(
    op: TranslationContext, self: ir.Value, other: ir.Value
) -> ir.Value:
    """matmul(Tensor(a) self, Tensor(b) other) -> Tensor(c)"""

    self_real = op.Gather(self, 0, axis=-1)
    self_imag = op.Gather(self, 1, axis=-1)
    other_real = op.Gather(other, 0, axis=-1)
    other_imag = op.Gather(other, 1, axis=-1)

    real1 = op.Concat(self_real, self_imag, axis=-1)
    imag1 = real1

    if (
        len(other.shape) == 2
    ):  # if other is a vector, we need to concatenate along the last dimension
        real2 = op.Concat(other_real, op.Neg(other_imag), axis=-1)
        imag2 = op.Concat(other_imag, other_real, axis=-1)
    else:
        real2 = op.Concat(other_real, op.Neg(other_imag), axis=-2)
        imag2 = op.Concat(other_imag, other_real, axis=-2)

    real = op.MatMul(real1, real2)
    imag = op.MatMul(imag1, imag2)

    return op.Concat(
        op.Unsqueeze(real, [-1]),
        op.Unsqueeze(imag, [-1]),
        axis=-1,
    )


@onnx_impl(aten.addmm.default)
def aten_addmm(
    op: TranslationContext,
    self: ir.Value,
    mat1: ir.Value,
    mat2: ir.Value,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> ir.Value:
    """addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"""

    # NOTE: ONNX Runtime does not support int inputs to Gemm as of 1.16.
    # To support int inputs, consider an overriding implementation that casts to float and back.

    alpha = float(alpha)
    beta = float(beta)

    # addmm only accepts 2d tensors: https://pytorch.org/docs/stable/generated/torch.addmm.html
    return op.Gemm(mat1, mat2, self, alpha=alpha, beta=beta)


@onnx_impl(aten.addbmm.default)
def aten_addbmm(
    op: TranslationContext,
    self: ir.Value,
    batch1: ir.Value,
    batch2: ir.Value,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> ir.Value:
    """addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor

    Performs a batch matrix-matrix product of matrices stored in `batch1` and `batch2`,
    with a reduced add step (all matrix multiplications get accumulated along the first
    dimension). `self` is added to the final result.

    `batch1` and `batch2` must be 3-D tensors each containing the same number of matrices.
    """
    builder = op.builder

    axes = builder.constant([0], dtype=ir.DataType.INT64)
    reduced_batches = op.ReduceSum(op.MatMul(batch1, batch2), axes, keepdims=False)
    if beta == 0:
        return op.Mul(reduced_batches, alpha)
    scaled_self = op.Mul(self, beta)
    return op.Add(scaled_self, op.Mul(reduced_batches, alpha))


@onnx_impl(aten.baddbmm.default)
def aten_baddbmm(
    op: TranslationContext,
    self: ir.Value,
    batch1: ir.Value,
    batch2: ir.Value,
    beta: ir.Value | None = None,
    alpha: ir.Value | None = None,
) -> ir.Value:
    """baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"""
    # beta and alpha can be SymFloat
    batch_mul = op.MatMul(batch1, batch2)
    if alpha is None or (not isinstance(alpha, ir.Value) and alpha == 1):
        mul_a = batch_mul
    else:
        mul_a = op.Mul(batch_mul, op.CastLike(alpha, self))
    if not isinstance(beta, ir.Value) and beta == 0:
        return mul_a
    if beta is None or (not isinstance(beta, ir.Value) and beta == 1):
        mul_b = self
    else:
        mul_b = op.Mul(self, op.CastLike(beta, self))
        if isinstance(beta, ir.Value):
            mul_b = op.Where(op.Equal(beta, 0), 0, mul_b)
    return op.Add(mul_a, mul_b)


@onnx_impl(aten.addmv.default)
def aten_addmv(
    op: TranslationContext,
    self: ir.Value,
    mat: ir.Value,
    vec: ir.Value,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> ir.Value:
    """addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor"""

    product = op.Mul(op.MatMul(mat, vec), alpha)
    if beta == 0:
        return product
    return op.Add(op.Mul(self, beta), product)


@onnx_impl(aten.addr.default)
def aten_addr(
    op: TranslationContext,
    self: ir.Value,
    vec1: ir.Value,
    vec2: ir.Value,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> ir.Value:
    """addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor

    Performs the outer-product of vectors vec1 and vec2 and adds it to the matrix input.
    """
    builder = op.builder
    vec1_shape = builder.constant([-1, 1], dtype=ir.DataType.INT64)
    vec2_shape = builder.constant([1, -1], dtype=ir.DataType.INT64)
    vec1_reshaped = op.Reshape(vec1, vec1_shape)
    vec2_reshaped = op.Reshape(vec2, vec2_shape)

    outer = op.MatMul(vec1_reshaped, vec2_reshaped)
    # https://github.com/pytorch/pytorch/blob/51664489ba6f6b2343bbec9af9ca99185e2a5dbc/aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp#L53-L54
    # When beta == 0, values in self should be ignored,
    # nans and infs in self should not propagate.
    alpha = op.CastLike(alpha, outer)
    if beta == 0.0:
        result = op.Mul(alpha, outer)
    else:
        beta = op.CastLike(beta, outer)
        result = op.Add(
            op.Mul(beta, self),
            op.Mul(alpha, outer),
        )

    return result


@onnx_impl(aten.einsum.default)
def aten_einsum(
    op: TranslationContext,
    equation: str,
    tensors: Sequence[ir.Value],
    path: int | None = None,
) -> ir.Value:
    """einsum(str equation, Tensor[] tensors, *, int[]? path=None) -> Tensor"""

    # Unpack the explicit tensor list into ONNX's variadic inputs.
    return op.Einsum(*tensors, equation=equation)


@onnx_impl(aten.contiguous.default)
def aten_contiguous(
    op: TranslationContext,
    self: ir.Value,
    memory_format: str = "contiguous_format",
) -> ir.Value:
    """contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)"""

    # ONNX does not have the notion of memory_format. It is always treated as a no-op.
    return op.Identity(self)


@onnx_impl(aten.copy.default)
def aten_copy(
    op: TranslationContext,
    self: ir.Value,
    src: ir.Value,
    non_blocking: bool = False,
) -> ir.Value:
    """copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor"""

    return op.CastLike(src, self)


@onnx_impl(aten.detach.default)
def aten_detach(op: TranslationContext, self: ir.Value) -> ir.Value:
    """detach(Tensor(a) self) -> Tensor(a)"""

    return op.Identity(self)


@onnx_impl(aten.lift_fresh_copy.default)
def aten_lift_fresh_copy(op: TranslationContext, self: ir.Value) -> ir.Value:
    """lift_fresh_copy(Tensor self) -> Tensor"""

    return op.Identity(self)


@onnx_impl(aten.resolve_conj.default)
def aten_resolve_conj(op: TranslationContext, self: ir.Value) -> ir.Value:
    """resolve_conj(Tensor(a) self) -> Tensor(a)"""

    return op.Identity(self)


@onnx_impl(aten.resolve_neg.default)
def aten_resolve_neg(op: TranslationContext, self: ir.Value) -> ir.Value:
    """resolve_neg(Tensor(a) self) -> Tensor(a)"""

    return op.Identity(self)


@onnx_impl(
    (aten._linalg_det.default, aten.linalg_det.default, aten.det.default),
    opset_introduced=18,
)
def aten_linalg_det(op: TranslationContext, A: ir.Value) -> ir.Value:
    """linalg_det(Tensor A) -> Tensor"""

    return op.Det(A)


@onnx_impl(aten.linalg_vector_norm.default, opset_introduced=18)
def aten_linalg_vector_norm(
    op: TranslationContext,
    self: ir.Value,
    ord: float = 2.0,
    dim: int | None = None,
    keepdim: bool = False,
    dtype: int = -1,
) -> ir.Value:
    """linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    builder = op.builder

    if dtype != -1:
        self = op.Cast(self, to=dtype)
    if dim is None:
        self = op.Reshape(self, builder.constant([-1], dtype=ir.DataType.INT64))
        keepdim = False
    else:
        if not isinstance(dim, ir.Value):
            dim = builder.constant(dim, dtype=ir.DataType.INT64)
        dim = op.Reshape(dim, builder.constant([-1], dtype=ir.DataType.INT64))

    if math.isinf(ord):
        self = op.Abs(self)
        if ord > 0:
            return op.ReduceMax(self, dim, keepdims=keepdim)
        else:
            return op.ReduceMin(self, dim, keepdims=keepdim)
    elif ord == 0.0:  # sum(x!=0) means count non-zero elements
        self_bool = op.Cast(self, to=ir.DataType.BOOL)
        self_0_1 = op.CastLike(self_bool, self)
        return op.ReduceSum(self_0_1, dim, keepdims=keepdim)
    elif ord == 1.0:
        return op.ReduceL1(self, dim, keepdims=keepdim)
    elif ord == 2.0:
        return op.ReduceL2(self, dim, keepdims=keepdim)
    else:
        if ord < 0 or ord % 2 != 0:
            # Not an even integer (could be odd, fractional or negative), use Abs
            self = op.Abs(self)
        self_pow = op.Pow(self, ord)
        exp = op.CastLike(1 / ord, self)
        return op.Pow(op.ReduceSum(self_pow, dim, keepdims=keepdim), exp)
