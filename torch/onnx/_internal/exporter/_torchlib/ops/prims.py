# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR prims translations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.onnx._internal.exporter._ir_builder import current_builder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx._internal.exporter._torchlib.ops import common as common_ops


if TYPE_CHECKING:
    from collections.abc import Sequence

    import onnx_ir as ir


prims = torch.ops.prims


@onnx_impl(prims.abs.default, opset_introduced=18)
def prims_abs(self: ir.Value) -> ir.Value:
    """abs(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Abs", self)[0]


@onnx_impl(prims.acos.default, opset_introduced=18)
def prims_acos(self: ir.Value) -> ir.Value:
    """acos(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Acos", self)[0]


@onnx_impl(prims.acosh.default, opset_introduced=18)
def prims_acosh(self: ir.Value) -> ir.Value:
    """acosh(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Acosh", self)[0]


@onnx_impl(prims.add.default, opset_introduced=18)
def prims_add(self: ir.Value, other: ir.Value) -> ir.Value:
    """add(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("Add", self, other)[0]


@onnx_impl(prims.asin.default, opset_introduced=18)
def prims_asin(self: ir.Value) -> ir.Value:
    """asin(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Asin", self)[0]


@onnx_impl(prims.asinh.default, opset_introduced=18)
def prims_asinh(self: ir.Value) -> ir.Value:
    """asinh(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Asinh", self)[0]


@onnx_impl(prims.atan.default, opset_introduced=18)
def prims_atan(self: ir.Value) -> ir.Value:
    """atan(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Atan", self)[0]


@onnx_impl(prims.atanh.default, opset_introduced=18)
def prims_atanh(self: ir.Value) -> ir.Value:
    """atanh(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Atanh", self)[0]


@onnx_impl(prims.broadcast_in_dim.default, opset_introduced=18)
def prims_broadcast_in_dim(
    a: ir.Value, shape: Sequence[ir.Value], broadcast_dimensions: Sequence[int]
) -> ir.Value:
    """broadcast_in_dim(Tensor(a) a, SymInt[] shape, int[] broadcast_dimensions) -> Tensor(a)"""
    builder = current_builder()

    target_rank = len(shape)

    if not broadcast_dimensions:
        # Special case: no broadcast dimensions - all target dims should be 1
        return builder.call("Expand", a, common_ops.merge_dims(shape))[0]

    # Create base shape of all 1s
    ones = [1] * target_rank

    # For each broadcast dimension, we'll replace the 1 with the actual input dimension
    # Since broadcast_dimensions is compile-time known, we can do this with individual operations
    intermediate_shape = ones

    for i, broadcast_dim in enumerate(broadcast_dimensions):
        # Get the input dimension value
        input_dim_value = builder.call("Shape", a, start=i, end=i + 1)[0]
        intermediate_shape[broadcast_dim] = input_dim_value

    # Reshape input to intermediate shape and expand to target
    reshaped = builder.call("Reshape", a, common_ops.merge_dims(intermediate_shape))[0]
    return builder.call("Expand", reshaped, common_ops.merge_dims(shape))[0]


@onnx_impl(prims.ceil.default, opset_introduced=18)
def prims_ceil(self: ir.Value) -> ir.Value:
    """ceil(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Ceil", self)[0]


@onnx_impl(prims.convert_element_type.default, opset_introduced=18)
def prims_convert_element_type(a: ir.Value, dtype: int) -> ir.Value:
    """convert_element_type(Tensor a, ScalarType dtype) -> Tensor"""

    # Different dtype branches are selected in Python while building the graph.
    return common_ops.cast_to(a, dtype)


@onnx_impl(prims.cos.default, opset_introduced=18)
def prims_cos(self: ir.Value) -> ir.Value:
    """cos(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Cos", self)[0]


@onnx_impl(prims.cosh.default, opset_introduced=18)
def prims_cosh(self: ir.Value) -> ir.Value:
    """cosh(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Cosh", self)[0]


@onnx_impl(prims.device_put.default, opset_introduced=18)
def prims_device_put(
    a: ir.Value,
    device: str = "unspecified",  # pylint: disable=unused-argument
) -> ir.Value:
    """device_put(Tensor a, Device device) -> Tensor"""
    builder = current_builder()

    # ONNX does not have the notion of a "device". So we just return the input
    return builder.call("Identity", a)[0]


@onnx_impl(prims.div.default, opset_introduced=18)
def prims_div(self: ir.Value, other: ir.Value) -> ir.Value:
    """div(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("Div", self, other)[0]


@onnx_impl(prims.eq.default, opset_introduced=18)
def prims_eq(self: ir.Value, other: ir.Value) -> ir.Value:
    """eq(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("Equal", self, other)[0]


@onnx_impl(prims.erf.default, opset_introduced=18)
def prims_erf(self: ir.Value) -> ir.Value:
    """erf(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Erf", self)[0]


@onnx_impl(prims.exp.default, opset_introduced=18)
def prims_exp(self: ir.Value) -> ir.Value:
    """exp(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Exp", self)[0]


@onnx_impl(prims.floor.default, opset_introduced=18)
def prims_floor(self: ir.Value) -> ir.Value:
    """floor(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Floor", self)[0]


@onnx_impl(prims.ge.default, opset_introduced=18)
def prims_ge(self: ir.Value, other: ir.Value) -> ir.Value:
    """ge(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("GreaterOrEqual", self, other)[0]


@onnx_impl(prims.gt.default, opset_introduced=18)
def prims_gt(self: ir.Value, other: ir.Value) -> ir.Value:
    """gt(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("Greater", self, other)[0]


@onnx_impl(prims.le.default, opset_introduced=18)
def prims_le(self: ir.Value, other: ir.Value) -> ir.Value:
    """le(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("LessOrEqual", self, other)[0]


@onnx_impl(prims.log.default, opset_introduced=18)
def prims_log(self: ir.Value) -> ir.Value:
    """log(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Log", self)[0]


@onnx_impl(prims.lt.default, opset_introduced=18)
def prims_lt(self: ir.Value, other: ir.Value) -> ir.Value:
    """lt(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("Less", self, other)[0]


@onnx_impl(prims.mul.default, opset_introduced=18)
def prims_mul(self: ir.Value, other: ir.Value) -> ir.Value:
    """mul(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("Mul", self, other)[0]


@onnx_impl(prims.ne.default, opset_introduced=18)
def prims_ne(self: ir.Value, other: ir.Value) -> ir.Value:
    """ne(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("Not", builder.call("Equal", self, other)[0])[0]


@onnx_impl(prims.neg.default, opset_introduced=18)
def prims_neg(self: ir.Value) -> ir.Value:
    """neg(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Neg", self)[0]


@onnx_impl(prims.pow.default, opset_introduced=18)
def prims_pow(self: ir.Value, other: ir.Value) -> ir.Value:
    """pow(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("Pow", self, other)[0]


@onnx_impl(prims.reshape.default, opset_introduced=18)
def prims_reshape(a: ir.Value, shape: ir.Value) -> ir.Value:
    """reshape(Tensor a, SymInt[] shape) -> Tensor"""
    builder = current_builder()

    return builder.call("Reshape", a, shape)[0]


@onnx_impl(prims.resize.default, opset_introduced=18)
def prims_resize(a: ir.Value, shape: ir.Value) -> ir.Value:
    """resize(Tensor a, SymInt[] shape) -> Tensor"""
    builder = current_builder()

    return builder.call("Expand", a, shape)[0]


@onnx_impl(prims.round.default, opset_introduced=18)
def prims_round(self: ir.Value) -> ir.Value:
    """round(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Round", self)[0]


@onnx_impl(prims.sin.default, opset_introduced=18)
def prims_sin(self: ir.Value) -> ir.Value:
    """sin(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Sin", self)[0]


@onnx_impl(prims.sinh.default, opset_introduced=18)
def prims_sinh(self: ir.Value) -> ir.Value:
    """sinh(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Sinh", self)[0]


@onnx_impl(prims.sqrt.default, opset_introduced=18)
def prims_sqrt(self: ir.Value) -> ir.Value:
    """sqrt(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Sqrt", self)[0]


@onnx_impl(prims.squeeze.default, opset_introduced=18)
def prims_squeeze(a: ir.Value, dimensions: Sequence[int]) -> ir.Value:
    """squeeze(Tensor(a) a, int[] dimensions) -> Tensor(a)"""
    builder = current_builder()

    return builder.call("Squeeze", a, dimensions)[0]


@onnx_impl(prims.sub.default, opset_introduced=18)
def prims_sub(self: ir.Value, other: ir.Value) -> ir.Value:
    """sub(Tensor self, Tensor other) -> Tensor"""
    builder = current_builder()

    return builder.call("Sub", self, other)[0]


@onnx_impl(prims.sum.default, opset_introduced=18)
def prims_sum(
    inp: ir.Value, dims: Sequence[int] | None, output_dtype: int | None = None
) -> ir.Value:
    """sum(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor"""
    builder = current_builder()

    result = builder.call("ReduceSum", inp, dims, keepdims=False)[0]

    if output_dtype is not None and output_dtype != -1:
        result = builder.call("Cast", result, to=output_dtype)[0]

    return result


@onnx_impl(prims.tan.default, opset_introduced=18)
def prims_tan(self: ir.Value) -> ir.Value:
    """tan(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Tan", self)[0]


@onnx_impl(prims.tanh.default, opset_introduced=18)
def prims_tanh(self: ir.Value) -> ir.Value:
    """tanh(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Tanh", self)[0]


@onnx_impl(prims.transpose.default, opset_introduced=18)
def prims_transpose(a: ir.Value, permutation: Sequence[int]) -> ir.Value:
    """transpose(Tensor(a) a, int[] permutation) -> Tensor(a)"""
    builder = current_builder()

    return builder.call("Transpose", a, perm=permutation)[0]


@onnx_impl(prims.var.default, opset_introduced=18)
def prims_var(
    inp: ir.Value,
    dims: Sequence[int] | None,
    correction: int,
    output_dtype: int | None = None,
) -> ir.Value:
    """var(Tensor inp, int[]? dims, *, int correction, ScalarType? output_dtype=None) -> Tensor"""
    builder = current_builder()

    if not dims:
        # dims can be empty in practice. We just use a None so it is not added in the ONNX graph
        dims = None
    sub_mean = builder.call(
        "Sub", inp, builder.call("ReduceMean", inp, dims, keepdims=True)[0]
    )[0]
    sqr_mean = builder.call("Mul", sub_mean, sub_mean)[0]
    var = builder.call("ReduceMean", sqr_mean, dims, keepdims=False)[0]
    # Adjust var according to correction value
    if correction != 0:
        inp_shape = builder.call("Shape", inp)[0]
        dim_size = (
            inp_shape
            if dims is None
            else builder.call("Gather", inp_shape, dims, axis=0)[0]
        )
        numel_float = builder.call(
            "CastLike", builder.call("ReduceProd", dim_size, keepdims=False)[0], inp
        )[0]
        mul = builder.call("Mul", var, numel_float)[0]
        # Subtract the correction value
        sub = builder.call(
            "Sub", numel_float, builder.call("CastLike", correction, inp)[0]
        )[0]
        var = builder.call("Div", mul, sub)[0]

    if output_dtype is not None and output_dtype != -1:
        var = builder.call("Cast", var, to=output_dtype)[0]

    return var


@onnx_impl(prims.where.default, opset_introduced=18)
def prims_where(pred: ir.Value, a: ir.Value, b: ir.Value) -> ir.Value:
    """where(Tensor pred, Tensor a, Tensor b) -> Tensor"""
    builder = current_builder()

    return builder.call("Where", pred, a, b)[0]
