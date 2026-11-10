# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR prims translations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx._internal.exporter._torchlib.ops import common as common_ops


if TYPE_CHECKING:
    from collections.abc import Sequence

    import onnx_ir as ir

    from torch.onnx._internal.exporter._ir_builder import TranslationContext


prims = torch.ops.prims


@onnx_impl(prims.abs.default, opset_introduced=18)
def prims_abs(op: TranslationContext, self: ir.Value) -> ir.Value:
    """abs(Tensor self) -> Tensor"""

    return op.Abs(self)


@onnx_impl(prims.acos.default, opset_introduced=18)
def prims_acos(op: TranslationContext, self: ir.Value) -> ir.Value:
    """acos(Tensor self) -> Tensor"""

    return op.Acos(self)


@onnx_impl(prims.acosh.default, opset_introduced=18)
def prims_acosh(op: TranslationContext, self: ir.Value) -> ir.Value:
    """acosh(Tensor self) -> Tensor"""

    return op.Acosh(self)


@onnx_impl(prims.add.default, opset_introduced=18)
def prims_add(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """add(Tensor self, Tensor other) -> Tensor"""

    return op.Add(self, other)


@onnx_impl(prims.asin.default, opset_introduced=18)
def prims_asin(op: TranslationContext, self: ir.Value) -> ir.Value:
    """asin(Tensor self) -> Tensor"""

    return op.Asin(self)


@onnx_impl(prims.asinh.default, opset_introduced=18)
def prims_asinh(op: TranslationContext, self: ir.Value) -> ir.Value:
    """asinh(Tensor self) -> Tensor"""

    return op.Asinh(self)


@onnx_impl(prims.atan.default, opset_introduced=18)
def prims_atan(op: TranslationContext, self: ir.Value) -> ir.Value:
    """atan(Tensor self) -> Tensor"""

    return op.Atan(self)


@onnx_impl(prims.atanh.default, opset_introduced=18)
def prims_atanh(op: TranslationContext, self: ir.Value) -> ir.Value:
    """atanh(Tensor self) -> Tensor"""

    return op.Atanh(self)


@onnx_impl(prims.broadcast_in_dim.default, opset_introduced=18)
def prims_broadcast_in_dim(
    op: TranslationContext,
    a: ir.Value,
    shape: Sequence[ir.Value],
    broadcast_dimensions: Sequence[int],
) -> ir.Value:
    """broadcast_in_dim(Tensor(a) a, SymInt[] shape, int[] broadcast_dimensions) -> Tensor(a)"""

    target_rank = len(shape)

    if not broadcast_dimensions:
        # Special case: no broadcast dimensions - all target dims should be 1
        return op.Expand(a, common_ops.merge_dims(op, shape))

    # Create base shape of all 1s
    ones = [1] * target_rank

    # For each broadcast dimension, we'll replace the 1 with the actual input dimension
    # Since broadcast_dimensions is compile-time known, we can do this with individual operations
    intermediate_shape = ones

    for i, broadcast_dim in enumerate(broadcast_dimensions):
        # Get the input dimension value
        input_dim_value = op.Shape(a, start=i, end=i + 1)
        intermediate_shape[broadcast_dim] = input_dim_value

    # Reshape input to intermediate shape and expand to target
    reshaped = op.Reshape(a, common_ops.merge_dims(op, intermediate_shape))
    return op.Expand(reshaped, common_ops.merge_dims(op, shape))


@onnx_impl(prims.ceil.default, opset_introduced=18)
def prims_ceil(op: TranslationContext, self: ir.Value) -> ir.Value:
    """ceil(Tensor self) -> Tensor"""

    return op.Ceil(self)


@onnx_impl(prims.convert_element_type.default, opset_introduced=18)
def prims_convert_element_type(
    op: TranslationContext, a: ir.Value, dtype: int
) -> ir.Value:
    """convert_element_type(Tensor a, ScalarType dtype) -> Tensor"""

    # Different dtype branches are selected in Python while building the graph.
    return common_ops.cast_to(op, a, dtype)


@onnx_impl(prims.cos.default, opset_introduced=18)
def prims_cos(op: TranslationContext, self: ir.Value) -> ir.Value:
    """cos(Tensor self) -> Tensor"""

    return op.Cos(self)


@onnx_impl(prims.cosh.default, opset_introduced=18)
def prims_cosh(op: TranslationContext, self: ir.Value) -> ir.Value:
    """cosh(Tensor self) -> Tensor"""

    return op.Cosh(self)


@onnx_impl(prims.device_put.default, opset_introduced=18)
def prims_device_put(
    op: TranslationContext,
    a: ir.Value,
    device: str = "unspecified",  # pylint: disable=unused-argument
) -> ir.Value:
    """device_put(Tensor a, Device device) -> Tensor"""

    # ONNX does not have the notion of a "device". So we just return the input
    return op.Identity(a)


@onnx_impl(prims.div.default, opset_introduced=18)
def prims_div(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """div(Tensor self, Tensor other) -> Tensor"""

    return op.Div(self, other)


@onnx_impl(prims.eq.default, opset_introduced=18)
def prims_eq(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """eq(Tensor self, Tensor other) -> Tensor"""

    return op.Equal(self, other)


@onnx_impl(prims.erf.default, opset_introduced=18)
def prims_erf(op: TranslationContext, self: ir.Value) -> ir.Value:
    """erf(Tensor self) -> Tensor"""

    return op.Erf(self)


@onnx_impl(prims.exp.default, opset_introduced=18)
def prims_exp(op: TranslationContext, self: ir.Value) -> ir.Value:
    """exp(Tensor self) -> Tensor"""

    return op.Exp(self)


@onnx_impl(prims.floor.default, opset_introduced=18)
def prims_floor(op: TranslationContext, self: ir.Value) -> ir.Value:
    """floor(Tensor self) -> Tensor"""

    return op.Floor(self)


@onnx_impl(prims.ge.default, opset_introduced=18)
def prims_ge(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """ge(Tensor self, Tensor other) -> Tensor"""

    return op.GreaterOrEqual(self, other)


@onnx_impl(prims.gt.default, opset_introduced=18)
def prims_gt(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """gt(Tensor self, Tensor other) -> Tensor"""

    return op.Greater(self, other)


@onnx_impl(prims.le.default, opset_introduced=18)
def prims_le(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """le(Tensor self, Tensor other) -> Tensor"""

    return op.LessOrEqual(self, other)


@onnx_impl(prims.log.default, opset_introduced=18)
def prims_log(op: TranslationContext, self: ir.Value) -> ir.Value:
    """log(Tensor self) -> Tensor"""

    return op.Log(self)


@onnx_impl(prims.lt.default, opset_introduced=18)
def prims_lt(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """lt(Tensor self, Tensor other) -> Tensor"""

    return op.Less(self, other)


@onnx_impl(prims.mul.default, opset_introduced=18)
def prims_mul(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """mul(Tensor self, Tensor other) -> Tensor"""

    return op.Mul(self, other)


@onnx_impl(prims.ne.default, opset_introduced=18)
def prims_ne(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """ne(Tensor self, Tensor other) -> Tensor"""

    return op.Not(op.Equal(self, other))


@onnx_impl(prims.neg.default, opset_introduced=18)
def prims_neg(op: TranslationContext, self: ir.Value) -> ir.Value:
    """neg(Tensor self) -> Tensor"""

    return op.Neg(self)


@onnx_impl(prims.pow.default, opset_introduced=18)
def prims_pow(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """pow(Tensor self, Tensor other) -> Tensor"""

    return op.Pow(self, other)


@onnx_impl(prims.reshape.default, opset_introduced=18)
def prims_reshape(op: TranslationContext, a: ir.Value, shape: ir.Value) -> ir.Value:
    """reshape(Tensor a, SymInt[] shape) -> Tensor"""

    return op.Reshape(a, shape)


@onnx_impl(prims.resize.default, opset_introduced=18)
def prims_resize(op: TranslationContext, a: ir.Value, shape: ir.Value) -> ir.Value:
    """resize(Tensor a, SymInt[] shape) -> Tensor"""

    return op.Expand(a, shape)


@onnx_impl(prims.round.default, opset_introduced=18)
def prims_round(op: TranslationContext, self: ir.Value) -> ir.Value:
    """round(Tensor self) -> Tensor"""

    return op.Round(self)


@onnx_impl(prims.sin.default, opset_introduced=18)
def prims_sin(op: TranslationContext, self: ir.Value) -> ir.Value:
    """sin(Tensor self) -> Tensor"""

    return op.Sin(self)


@onnx_impl(prims.sinh.default, opset_introduced=18)
def prims_sinh(op: TranslationContext, self: ir.Value) -> ir.Value:
    """sinh(Tensor self) -> Tensor"""

    return op.Sinh(self)


@onnx_impl(prims.sqrt.default, opset_introduced=18)
def prims_sqrt(op: TranslationContext, self: ir.Value) -> ir.Value:
    """sqrt(Tensor self) -> Tensor"""

    return op.Sqrt(self)


@onnx_impl(prims.squeeze.default, opset_introduced=18)
def prims_squeeze(
    op: TranslationContext, a: ir.Value, dimensions: Sequence[int]
) -> ir.Value:
    """squeeze(Tensor(a) a, int[] dimensions) -> Tensor(a)"""

    return op.Squeeze(a, dimensions)


@onnx_impl(prims.sub.default, opset_introduced=18)
def prims_sub(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """sub(Tensor self, Tensor other) -> Tensor"""

    return op.Sub(self, other)


@onnx_impl(prims.sum.default, opset_introduced=18)
def prims_sum(
    op: TranslationContext,
    inp: ir.Value,
    dims: Sequence[int] | None,
    output_dtype: int | None = None,
) -> ir.Value:
    """sum(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor"""

    result = op.ReduceSum(inp, dims, keepdims=False)

    if output_dtype is not None and output_dtype != -1:
        result = op.Cast(result, to=output_dtype)

    return result


@onnx_impl(prims.tan.default, opset_introduced=18)
def prims_tan(op: TranslationContext, self: ir.Value) -> ir.Value:
    """tan(Tensor self) -> Tensor"""

    return op.Tan(self)


@onnx_impl(prims.tanh.default, opset_introduced=18)
def prims_tanh(op: TranslationContext, self: ir.Value) -> ir.Value:
    """tanh(Tensor self) -> Tensor"""

    return op.Tanh(self)


@onnx_impl(prims.transpose.default, opset_introduced=18)
def prims_transpose(
    op: TranslationContext, a: ir.Value, permutation: Sequence[int]
) -> ir.Value:
    """transpose(Tensor(a) a, int[] permutation) -> Tensor(a)"""

    return op.Transpose(a, perm=permutation)


@onnx_impl(prims.var.default, opset_introduced=18)
def prims_var(
    op: TranslationContext,
    inp: ir.Value,
    dims: Sequence[int] | None,
    correction: int,
    output_dtype: int | None = None,
) -> ir.Value:
    """var(Tensor inp, int[]? dims, *, int correction, ScalarType? output_dtype=None) -> Tensor"""

    if not dims:
        # dims can be empty in practice. We just use a None so it is not added in the ONNX graph
        dims = None
    sub_mean = op.Sub(inp, op.ReduceMean(inp, dims, keepdims=True))
    sqr_mean = op.Mul(sub_mean, sub_mean)
    var = op.ReduceMean(sqr_mean, dims, keepdims=False)
    # Adjust var according to correction value
    if correction != 0:
        inp_shape = op.Shape(inp)
        dim_size = inp_shape if dims is None else op.Gather(inp_shape, dims, axis=0)
        numel_float = op.CastLike(op.ReduceProd(dim_size, keepdims=False), inp)
        mul = op.Mul(var, numel_float)
        # Subtract the correction value
        sub = op.Sub(numel_float, op.CastLike(correction, inp))
        var = op.Div(mul, sub)

    if output_dtype is not None and output_dtype != -1:
        var = op.Cast(var, to=output_dtype)

    return var


@onnx_impl(prims.where.default, opset_introduced=18)
def prims_where(
    op: TranslationContext, pred: ir.Value, a: ir.Value, b: ir.Value
) -> ir.Value:
    """where(Tensor pred, Tensor a, Tensor b) -> Tensor"""

    return op.Where(pred, a, b)
