# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR linalg translations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._ir_builder import current_builder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


if TYPE_CHECKING:
    from collections.abc import Sequence


aten = torch.ops.aten


@onnx_impl(aten.mm.default)
def aten_mm(self: ir.Value, mat2: ir.Value) -> ir.Value:
    """mm(Tensor self, Tensor mat2) -> Tensor"""
    builder = current_builder()

    return builder.call("MatMul", self, mat2)[0]


@onnx_impl(aten.bmm.default)
def aten_bmm(self: ir.Value, mat2: ir.Value) -> ir.Value:
    """bmm(Tensor self, Tensor mat2) -> Tensor"""
    builder = current_builder()

    return builder.call("MatMul", self, mat2)[0]


@onnx_impl(aten.mv.default)
def aten_mv(self: ir.Value, vec: ir.Value) -> ir.Value:
    """mv(Tensor self, Tensor vec) -> Tensor"""
    builder = current_builder()

    return builder.call("MatMul", self, vec)[0]


@onnx_impl(aten.dot.default)
def aten_dot(self: ir.Value, tensor: ir.Value) -> ir.Value:
    """dot(Tensor self, Tensor tensor) -> Tensor"""
    builder = current_builder()

    return builder.call("MatMul", self, tensor)[0]


@onnx_impl(aten.matmul.default)
def aten_matmul(self: ir.Value, other: ir.Value) -> ir.Value:
    """matmul(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("MatMul", self, other)[0]


@onnx_impl(aten.matmul.default, complex=True)
def aten_matmul_complex(self: ir.Value, other: ir.Value) -> ir.Value:
    """matmul(Tensor(a) self, Tensor(b) other) -> Tensor(c)"""
    builder = current_builder()

    self_real = builder.call("Gather", self, 0, axis=-1)[0]
    self_imag = builder.call("Gather", self, 1, axis=-1)[0]
    other_real = builder.call("Gather", other, 0, axis=-1)[0]
    other_imag = builder.call("Gather", other, 1, axis=-1)[0]

    real1 = builder.call("Concat", self_real, self_imag, axis=-1)[0]
    imag1 = real1

    if (
        len(other.shape) == 2
    ):  # if other is a vector, we need to concatenate along the last dimension
        real2 = builder.call(
            "Concat", other_real, builder.call("Neg", other_imag)[0], axis=-1
        )[0]
        imag2 = builder.call("Concat", other_imag, other_real, axis=-1)[0]
    else:
        real2 = builder.call(
            "Concat", other_real, builder.call("Neg", other_imag)[0], axis=-2
        )[0]
        imag2 = builder.call("Concat", other_imag, other_real, axis=-2)[0]

    real = builder.call("MatMul", real1, real2)[0]
    imag = builder.call("MatMul", imag1, imag2)[0]

    return builder.call(
        "Concat",
        builder.call("Unsqueeze", real, [-1])[0],
        builder.call("Unsqueeze", imag, [-1])[0],
        axis=-1,
    )[0]


@onnx_impl(aten.addmm.default)
def aten_addmm(
    self: ir.Value,
    mat1: ir.Value,
    mat2: ir.Value,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> ir.Value:
    """addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"""
    builder = current_builder()

    # NOTE: ONNX Runtime does not support int inputs to Gemm as of 1.16.
    # To support int inputs, consider an overriding implementation that casts to float and back.

    alpha = float(alpha)
    beta = float(beta)

    # addmm only accepts 2d tensors: https://pytorch.org/docs/stable/generated/torch.addmm.html
    return builder.call("Gemm", mat1, mat2, self, alpha=alpha, beta=beta)[0]


@onnx_impl(aten.addbmm.default)
def aten_addbmm(
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
    builder = current_builder()

    axes = builder.constant([0], dtype=ir.DataType.INT64)
    reduced_batches = builder.call(
        "ReduceSum", builder.call("MatMul", batch1, batch2)[0], axes, keepdims=False
    )[0]
    if beta == 0:
        return builder.call("Mul", reduced_batches, alpha)[0]
    scaled_self = builder.call("Mul", self, beta)[0]
    return builder.call(
        "Add", scaled_self, builder.call("Mul", reduced_batches, alpha)[0]
    )[0]


@onnx_impl(aten.baddbmm.default)
def aten_baddbmm(
    self: ir.Value,
    batch1: ir.Value,
    batch2: ir.Value,
    beta: ir.Value | None = None,
    alpha: ir.Value | None = None,
) -> ir.Value:
    """baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor"""
    builder = current_builder()
    # beta and alpha can be SymFloat
    batch_mul = builder.call("MatMul", batch1, batch2)[0]
    if alpha is None or (not isinstance(alpha, ir.Value) and alpha == 1):
        mul_a = batch_mul
    else:
        mul_a = builder.call(
            "Mul", batch_mul, builder.call("CastLike", alpha, self)[0]
        )[0]
    if not isinstance(beta, ir.Value) and beta == 0:
        return mul_a
    if beta is None or (not isinstance(beta, ir.Value) and beta == 1):
        mul_b = self
    else:
        mul_b = builder.call("Mul", self, builder.call("CastLike", beta, self)[0])[0]
        if isinstance(beta, ir.Value):
            mul_b = builder.call("Where", builder.call("Equal", beta, 0)[0], 0, mul_b)[
                0
            ]
    return builder.call("Add", mul_a, mul_b)[0]


@onnx_impl(aten.addmv.default)
def aten_addmv(
    self: ir.Value, mat: ir.Value, vec: ir.Value, beta: float = 1.0, alpha: float = 1.0
) -> ir.Value:
    """addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor"""
    builder = current_builder()

    product = builder.call("Mul", builder.call("MatMul", mat, vec)[0], alpha)[0]
    if beta == 0:
        return product
    return builder.call("Add", builder.call("Mul", self, beta)[0], product)[0]


@onnx_impl(aten.addr.default)
def aten_addr(
    self: ir.Value,
    vec1: ir.Value,
    vec2: ir.Value,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> ir.Value:
    """addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor

    Performs the outer-product of vectors vec1 and vec2 and adds it to the matrix input.
    """
    builder = current_builder()
    vec1_shape = builder.constant([-1, 1], dtype=ir.DataType.INT64)
    vec2_shape = builder.constant([1, -1], dtype=ir.DataType.INT64)
    vec1_reshaped = builder.call("Reshape", vec1, vec1_shape)[0]
    vec2_reshaped = builder.call("Reshape", vec2, vec2_shape)[0]

    outer = builder.call("MatMul", vec1_reshaped, vec2_reshaped)[0]
    # https://github.com/pytorch/pytorch/blob/51664489ba6f6b2343bbec9af9ca99185e2a5dbc/aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp#L53-L54
    # When beta == 0, values in self should be ignored,
    # nans and infs in self should not propagate.
    alpha = builder.call("CastLike", alpha, outer)[0]
    if beta == 0.0:
        result = builder.call("Mul", alpha, outer)[0]
    else:
        beta = builder.call("CastLike", beta, outer)[0]
        result = builder.call(
            "Add",
            builder.call("Mul", beta, self)[0],
            builder.call("Mul", alpha, outer)[0],
        )[0]

    return result


@onnx_impl(aten.einsum.default)
def aten_einsum(
    equation: str,
    tensors: Sequence[ir.Value],
    path: int | None = None,
) -> ir.Value:
    """einsum(str equation, Tensor[] tensors, *, int[]? path=None) -> Tensor"""
    builder = current_builder()

    # Unpack the explicit tensor list into ONNX's variadic inputs.
    return builder.call("Einsum", *tensors, equation=equation)[0]


@onnx_impl(aten.contiguous.default)
def aten_contiguous(
    self: ir.Value,
    memory_format: str = "contiguous_format",
) -> ir.Value:
    """contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)"""
    builder = current_builder()

    # ONNX does not have the notion of memory_format. It is always treated as a no-op.
    return builder.call("Identity", self)[0]


@onnx_impl(aten.copy.default)
def aten_copy(
    self: ir.Value,
    src: ir.Value,
    non_blocking: bool = False,
) -> ir.Value:
    """copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor"""
    builder = current_builder()

    return builder.call("CastLike", src, self)[0]


@onnx_impl(aten.detach.default)
def aten_detach(self: ir.Value) -> ir.Value:
    """detach(Tensor(a) self) -> Tensor(a)"""
    builder = current_builder()

    return builder.call("Identity", self)[0]


@onnx_impl(aten.lift_fresh_copy.default)
def aten_lift_fresh_copy(self: ir.Value) -> ir.Value:
    """lift_fresh_copy(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Identity", self)[0]


@onnx_impl(aten.resolve_conj.default)
def aten_resolve_conj(self: ir.Value) -> ir.Value:
    """resolve_conj(Tensor(a) self) -> Tensor(a)"""
    builder = current_builder()

    return builder.call("Identity", self)[0]


@onnx_impl(aten.resolve_neg.default)
def aten_resolve_neg(self: ir.Value) -> ir.Value:
    """resolve_neg(Tensor(a) self) -> Tensor(a)"""
    builder = current_builder()

    return builder.call("Identity", self)[0]


@onnx_impl(
    (aten._linalg_det.default, aten.linalg_det.default, aten.det.default),
    opset_introduced=18,
)
def aten_linalg_det(A: ir.Value) -> ir.Value:
    """linalg_det(Tensor A) -> Tensor"""
    builder = current_builder()

    return builder.call("Det", A)[0]


@onnx_impl(aten.linalg_vector_norm.default, opset_introduced=18)
def aten_linalg_vector_norm(
    self: ir.Value,
    ord: float = 2.0,
    dim: int | None = None,
    keepdim: bool = False,
    dtype: int = -1,
) -> ir.Value:
    """linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"""
    builder = current_builder()

    if dtype != -1:
        self = builder.call("Cast", self, to=dtype)[0]
    if dim is None:
        self = builder.call(
            "Reshape", self, builder.constant([-1], dtype=ir.DataType.INT64)
        )[0]
        keepdim = False
    else:
        if not isinstance(dim, ir.Value):
            dim = builder.constant(dim, dtype=ir.DataType.INT64)
        dim = builder.call(
            "Reshape", dim, builder.constant([-1], dtype=ir.DataType.INT64)
        )[0]

    if math.isinf(ord):
        self = builder.call("Abs", self)[0]
        if ord > 0:
            return builder.call("ReduceMax", self, dim, keepdims=keepdim)[0]
        else:
            return builder.call("ReduceMin", self, dim, keepdims=keepdim)[0]
    elif ord == 0.0:  # sum(x!=0) means count non-zero elements
        self_bool = builder.call("Cast", self, to=ir.DataType.BOOL)[0]
        self_0_1 = builder.call("CastLike", self_bool, self)[0]
        return builder.call("ReduceSum", self_0_1, dim, keepdims=keepdim)[0]
    elif ord == 1.0:
        return builder.call("ReduceL1", self, dim, keepdims=keepdim)[0]
    elif ord == 2.0:
        return builder.call("ReduceL2", self, dim, keepdims=keepdim)[0]
    else:
        if ord < 0 or ord % 2 != 0:
            # Not an even integer (could be odd, fractional or negative), use Abs
            self = builder.call("Abs", self)[0]
        self_pow = builder.call("Pow", self, ord)[0]
        exp = builder.call("CastLike", 1 / ord, self)[0]
        return builder.call(
            "Pow", builder.call("ReduceSum", self_pow, dim, keepdims=keepdim)[0], exp
        )[0]
