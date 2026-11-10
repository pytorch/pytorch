# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR residual translations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._ir_builder import GraphBuilder, TranslationContext
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx._internal.exporter._torchlib.ops import common as common_ops, core


if TYPE_CHECKING:
    from collections.abc import Sequence


aten = torch.ops.aten


@onnx_impl(aten._log_softmax.default)
def aten__log_softmax(
    op: TranslationContext, self: ir.Value, dim: int, half_to_float: bool
) -> ir.Value:
    """_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"""
    builder = op.builder

    self_is_scalar = len(self.shape) == 0
    if half_to_float and self.dtype in {ir.DataType.FLOAT16, ir.DataType.BFLOAT16}:
        self = op.Cast(self, to=ir.DataType.FLOAT)
    if self_is_scalar:
        self = op.Unsqueeze(self, builder.constant([0], dtype=ir.DataType.INT64))
    result = op.LogSoftmax(self, axis=dim)
    if self_is_scalar:
        result = op.Squeeze(result, builder.constant([0], dtype=ir.DataType.INT64))
    return result


@onnx_impl(aten._softmax.default)
def aten__softmax(
    op: TranslationContext, self: ir.Value, dim: int, half_to_float: bool
) -> ir.Value:
    """_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"""
    builder = op.builder

    self_is_scalar = len(self.shape) == 0

    if half_to_float and self.dtype in {ir.DataType.FLOAT16, ir.DataType.BFLOAT16}:
        self = op.Cast(self, to=ir.DataType.FLOAT)

    if self_is_scalar:
        self = op.Unsqueeze(self, builder.constant([0], dtype=ir.DataType.INT64))
    result = op.Softmax(self, axis=dim)
    if self_is_scalar:
        # Convert to scalar when input is scalar
        result = op.Squeeze(result)

    return result


@onnx_impl(aten.allclose.default)
def aten_allclose(
    op: TranslationContext,
    self: ir.Value,
    other: ir.Value,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> ir.Value:
    """allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool"""

    # FIXME: check equal_nan when self and other are all NaN
    # |input - other| <= atol + rtol x |other|
    left_part = op.Abs(op.Sub(self, other))
    right_part = op.Add(atol, op.Mul(rtol, op.Abs(other)))
    is_close = op.LessOrEqual(left_part, right_part)
    is_close_int = op.Cast(is_close, to=ir.DataType.INT8)

    # If min is 0, some elements are not close -> allclose is False
    # If min is 1, all elements are close -> allclose is True
    return op.Cast(
        op.ReduceMin(is_close_int, keepdims=False),
        to=ir.DataType.BOOL,
    )


@onnx_impl(aten.bilinear.default)
def aten_bilinear(
    op: TranslationContext,
    input1: ir.Value,
    input2: ir.Value,
    weight: ir.Value,
    bias: ir.Value | None = None,
) -> ir.Value:
    """bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias=None) -> Tensor"""
    builder = op.builder

    # Bilinear transformation: y = x1^T A x2 + b
    # input1 shape: (..., in1_features)
    # input2 shape: (..., in2_features)
    # weight shape: (out_features, in1_features, in2_features)
    # bias shape: (out_features) - optional
    # output shape: (..., out_features)

    # input1 and input2 must have identical batch dimensions
    # Use MatMul to compute the bilinear transformation
    batch_size = op.Shape(input1, start=0, end=-1)
    input1_shape = op.Shape(input1, start=-1)
    input2_shape = op.Shape(input2, start=-1)
    output_shape = op.Shape(weight, start=0, end=1)
    neg_1 = builder.constant([-1], dtype=ir.DataType.INT64)

    # (out_features, in1_features, in2_features) -> (in1_features, out_features, in2_features)
    W_permute = op.Transpose(weight, perm=[1, 0, 2])

    # (in1_features, out_features, in2_features) -> (in1_features, out_features * in2_features)
    W_flat = op.Reshape(
        W_permute,
        op.Concat(
            input1_shape,
            op.Mul(output_shape, input2_shape),
            axis=0,
        ),
    )

    # (..., in1_features) @ (in1_features, out_features * in2_features) -> (..., out_features * in2_features)
    tmp = op.MatMul(input1, W_flat)

    # (..., out_features * in2_features) -> (..., out_features, in2_features)
    tmp = op.Reshape(
        tmp,
        op.Concat(batch_size, output_shape, input2_shape, axis=0),
    )

    # (..., in2_features) -> (..., in2_features, 1)
    #   -> (..., out_features, in2_features) @ (..., in2_features, 1)
    #   -> (..., out_features, 1) -> (..., out_features)
    result = op.Squeeze(
        op.MatMul(tmp, op.Unsqueeze(input2, neg_1)),
        neg_1,
    )

    if bias is not None:
        result = op.Add(result, bias)

    return result


@onnx_impl(aten.bincount.default)
def aten_bincount(
    op: TranslationContext,
    self: ir.Value,
    weights: ir.Value | None = None,
    minlength: int = 0,
) -> ir.Value:
    """bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor

    ``weights`` is not supported. Negative inputs are rejected by torch and are not
    handled here (ONNX integer ops would wrap them around).
    """
    builder = op.builder
    if weights is not None:
        raise NotImplementedError("aten::bincount with weights is not supported.")

    self = op.Cast(self, to=ir.DataType.INT64)
    axis_0 = builder.constant([0], dtype=ir.DataType.INT64)
    # Append a 0 so ReduceMax is defined even when ``self`` is empty. It only sizes the
    # output and never contributes to the counts (the scatter below uses ``self``).
    data_max = op.Unsqueeze(
        op.ReduceMax(
            op.Concat(self, builder.constant([0], dtype=ir.DataType.INT64), axis=0),
            keepdims=0,
        ),
        axis_0,
    )
    # An empty input yields depth 0, so the output is empty unless ``minlength`` applies.
    non_empty = op.Unsqueeze(
        op.Cast(
            op.Greater(
                op.Size(self),
                builder.constant(0, dtype=ir.DataType.INT64),
            ),
            to=ir.DataType.INT64,
        ),
        axis_0,
    )
    depth = op.Mul(
        op.Add(data_max, builder.constant([1], dtype=ir.DataType.INT64)),
        non_empty,
    )
    if minlength > 0:
        depth = op.Max(depth, builder.constant([minlength], dtype=ir.DataType.INT64))

    # Scatter-add 1 for each value into a zero vector of length ``depth``. This uses
    # O(N + depth) memory instead of the dense O(N * depth) one-hot, and behaves
    # correctly for empty inputs.
    zeros = op.Expand(builder.constant(0, dtype=ir.DataType.INT64), depth)
    ones = op.Expand(
        builder.constant(1, dtype=ir.DataType.INT64),
        op.Shape(self),
    )
    return op.ScatterElements(zeros, self, ones, axis=0, reduction="add")


@onnx_impl(aten.complex.default)
def aten_complex(op: TranslationContext, real: ir.Value, imag: ir.Value) -> ir.Value:
    """complex(Tensor real, Tensor imag) -> Tensor"""

    # Broadcast the real and imaginary parts to the same shape
    broadcasted_shape = core._shape_of_broadcast_tensors(op, real, imag)
    real = op.Expand(real, broadcasted_shape)
    imag = op.Expand(imag, broadcasted_shape)

    return op.Concat(
        op.Unsqueeze(real, [-1]),
        op.Unsqueeze(imag, [-1]),
        axis=-1,
    )


@onnx_impl(aten.constant_pad_nd.default)
def aten_constant_pad_nd(
    op: TranslationContext, self: ir.Value, pad: Sequence[ir.Value], value: float = 0.0
) -> ir.Value:
    """constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor"""

    # The desired order of paddings is
    # dim_0_begin, dim_1_begin, ... , dim_0_end, ..., dim_n_end.
    # n is the dimension of input.
    # assume zero-dimensions in the beginning
    rank = len(self.shape)
    paddings = list(pad) + [0] * (rank * 2 - len(pad))
    # reverse order and collate first beginnings and then ends
    paddings = paddings[-2::-2] + paddings[-1::-2]
    constant_value = op.Constant(value=ir.tensor(value, dtype=self.dtype))

    return op.Pad(self, common_ops.merge_dims(op, paddings), constant_value)


@onnx_impl((aten.cross.default, aten.linalg_cross.default))
def aten_cross(
    op: TranslationContext, self: ir.Value, other: ir.Value, dim: int = -1
) -> ir.Value:
    """cross(Tensor self, Tensor other, int? dim=None) -> Tensor"""

    # Reference https://en.wikipedia.org/w/index.php?title=Cross_product&oldid=1143125073
    a1, a2, a3 = op.Split(self, axis=dim, num_outputs=3, outputs=3)
    b1, b2, b3 = op.Split(other, axis=dim, num_outputs=3, outputs=3)
    # Broadcasting is implicitly supported by Mul
    c1 = op.Sub(op.Mul(a2, b3), op.Mul(a3, b2))
    c2 = op.Sub(op.Mul(a3, b1), op.Mul(a1, b3))
    c3 = op.Sub(op.Mul(a1, b2), op.Mul(a2, b1))

    return op.Concat(c1, c2, c3, axis=dim)


@onnx_impl(aten.dropout.default)
def aten_dropout(
    op: TranslationContext, input: ir.Value, p: float | ir.Value, train: bool | ir.Value
) -> ir.Value:
    """dropout(Tensor input, float p, bool train) -> Tensor"""
    builder = op.builder
    # ONNX Dropout requires ratio < 1; PyTorch also supports dropping every element.
    if isinstance(p, (int, float)) and p == 1 and train is True:
        return op.Mul(input, op.CastLike(0, input))

    if input.shape is not None and len(input.shape) == 0:
        input = op.Reshape(input, builder.constant([-1], dtype=ir.DataType.INT64))
        result, _ = op.Dropout(input, p, train, outputs=2)
        result = op.Squeeze(result)
    else:
        result, _ = op.Dropout(input, p, train, outputs=2)

    return result


@onnx_impl(aten.equal.default)
def aten_equal(op: TranslationContext, self: ir.Value, other: ir.Value) -> ir.Value:
    """equal(Tensor self, Tensor other) -> bool"""

    # NOTE: Torch aten::equal returns a single Boolean while ONNX Equal is elementwise.
    # The equivalent Torch op with ONNX Equal is aten::eq.
    elementwise_equal = op.Equal(self, other)
    elementwise_equal_int = op.Cast(elementwise_equal, to=ir.DataType.INT64)
    # ReduceMin does not support bool. So we cast to int64
    all_equal = op.ReduceMin(elementwise_equal_int, keepdims=False)
    return op.Cast(all_equal, to=ir.DataType.BOOL)


@onnx_impl(aten.fake_quantize_per_channel_affine.default)
def aten_fake_quantize_per_channel_affine(
    op: TranslationContext,
    self: ir.Value,
    scale: ir.Value,  # float32 specifically!
    zero_point: ir.Value,  # int32, float32 or float16 only!
    axis: int,
    quant_min: int,
    quant_max: int,
) -> ir.Value:
    """fake_quantize_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> Tensor"""

    # NOTE: (0, 127) is allowed as special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    if (quant_min, quant_max) not in [(0, 255), (-128, 127), (0, 127)]:
        raise NotImplementedError(
            "For (quant_min, quant_max), ONNX allows only "
            "(0, 127), (0, 255) and (-128, 127). "
            f"Got ({quant_min}, {quant_max})",
        )

    if quant_min == 0:
        int_dtype = ir.DataType.UINT8
    else:
        int_dtype = ir.DataType.INT8

    # TODO: When opset >= 19, remove this cast
    orig_dtype = self.type.dtype
    if self.type.dtype not in {ir.DataType.FLOAT, ir.DataType.INT32}:
        self = op.Cast(self, to=ir.DataType.FLOAT)

    if zero_point.type.dtype == ir.DataType.INT32:
        zero_point = op.Cast(zero_point, to=int_dtype)
    else:
        raise NotImplementedError(
            "ONNX only supports integer values for the zero_point parameter. "
            f"Got {zero_point.type.dtype}",
        )

    quantized = op.QuantizeLinear(self, scale, zero_point, axis=axis)

    # See comment about, PyTorch-specific (0, 127) handling
    if (quant_min, quant_max) == (0, 127):
        const_127 = op.Cast(127, to=int_dtype)
        quantized = op.Clip(quantized, None, const_127)

    output = op.DequantizeLinear(quantized, scale, zero_point, axis=axis)

    # TODO: When opset >= 23, remove this cast and set output_dtype on DequantizeLinear
    if orig_dtype != ir.DataType.FLOAT:
        output = op.Cast(output, to=orig_dtype)

    return output


@onnx_impl(aten.fake_quantize_per_tensor_affine.default)
def aten_fake_quantize_per_tensor_affine(
    op: TranslationContext,
    self: ir.Value,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
) -> ir.Value:
    """fake_quantize_per_tensor_affine(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> Tensor"""

    return _aten_fake_quantize_per_tensor_affine(
        op, self, scale, zero_point, quant_min, quant_max
    )


@onnx_impl(aten.fake_quantize_per_tensor_affine.tensor_qparams)
def aten_fake_quantize_per_tensor_affine_tensor_qparams(
    op: TranslationContext,
    self: ir.Value,
    scale: ir.Value,
    zero_point: ir.Value,
    quant_min: int,
    quant_max: int,
) -> ir.Value:
    """fake_quantize_per_tensor_affine(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max) -> Tensor"""

    return _aten_fake_quantize_per_tensor_affine(
        op, self, scale, zero_point, quant_min, quant_max
    )


@onnx_impl(aten.grid_sampler.default)
def aten_grid_sampler(
    op: TranslationContext,
    input: ir.Value,
    grid: ir.Value,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> ir.Value:
    """grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor"""
    builder = op.builder

    inter_mode_options = (
        ("linear", "nearest", "cubic")
        if builder.opset >= 20
        else ("bilinear", "nearest", "bicubic")
    )
    inter_mode_str = inter_mode_options[interpolation_mode]

    padding_mode_options = ("zeros", "border", "reflection")
    padding_mode_str = padding_mode_options[padding_mode]

    return op.GridSample(
        input,
        grid,
        align_corners=align_corners,
        mode=inter_mode_str,
        padding_mode=padding_mode_str,
    )


@onnx_impl(aten.grid_sampler_2d.default)
def aten_grid_sampler_2d(
    op: TranslationContext,
    input: ir.Value,
    grid: ir.Value,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> ir.Value:
    """grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor"""
    builder = op.builder

    inter_mode_options = (
        ("linear", "nearest", "cubic")
        if builder.opset >= 20
        else ("bilinear", "nearest", "bicubic")
    )
    inter_mode_str = inter_mode_options[interpolation_mode]

    padding_mode_options = ("zeros", "border", "reflection")
    padding_mode_str = padding_mode_options[padding_mode]

    return op.GridSample(
        input,
        grid,
        align_corners=align_corners,
        mode=inter_mode_str,
        padding_mode=padding_mode_str,
    )


@onnx_impl(aten._grouped_mm.default)
def aten_grouped_mm(
    op: TranslationContext,
    self: ir.Value,
    mat2: ir.Value,
    offs: ir.Value | None = None,
    bias: ir.Value | None = None,
    out_dtype: int = -1,
) -> ir.Value:
    """_grouped_mm(Tensor self, Tensor mat2, *, Tensor? offs=None, Tensor? bias=None, int? out_dtype=None) -> Tensor"""
    builder = op.builder

    if offs is not None:
        if self.shape is None or mat2.shape is None:
            raise NotImplementedError("Grouped matmul requires known operand ranks.")
        a_rank, b_rank = len(self.shape), len(mat2.shape)
        if a_rank not in (2, 3) or b_rank not in (2, 3) or (a_rank == b_rank == 3):
            raise ValueError(
                "Grouped matmul with offsets requires at least one 2D operand."
            )
        if offs.shape is None or len(offs.shape) != 1:
            raise ValueError("Grouped matmul offsets must be 1D.")
        groups = offs.shape[0]
        if not isinstance(groups, int):
            raise NotImplementedError(
                "Grouped matmul requires a statically known number of groups."
            )
        for shape in (self.shape, mat2.shape):
            if len(shape) == 3 and isinstance(shape[0], int) and shape[0] != groups:
                raise ValueError(
                    "Grouped matmul offsets and operand group counts must match."
                )
        if bias is not None:
            raise NotImplementedError(
                "Grouped matmul with offsets does not support bias."
            )
        if out_dtype is not None and out_dtype != -1 and out_dtype != self.dtype:
            raise NotImplementedError(
                "Grouped matmul with offsets requires the output dtype to match the input."
            )

        a_is_2d, b_is_2d = a_rank == 2, b_rank == 2
        if groups == 0:
            output_shape = op.Concat(
                op.Shape(self, start=a_rank - 2, end=a_rank - 1),
                op.Shape(mat2, start=b_rank - 1, end=b_rank),
                axis=0,
            )
            if a_is_2d and b_is_2d:
                output_shape = op.Concat(
                    builder.constant([0], dtype=ir.DataType.INT64),
                    output_shape,
                    axis=0,
                )
            return op.Expand(op.CastLike(0, self), output_shape)

        # Only the group count is static. Boundaries remain values in the graph.
        ends = op.Cast(offs, to=ir.DataType.INT64)
        start = builder.constant([0], dtype=ir.DataType.INT64)
        outputs = []
        for i in range(groups):
            end = op.Gather(ends, [i], axis=0)
            a = (
                op.Slice(self, start, end, [1 if b_is_2d else 0])
                if a_is_2d
                else op.Gather(self, i, axis=0)
            )
            b = (
                op.Slice(mat2, start, end, [0 if a_is_2d else 1])
                if b_is_2d
                else op.Gather(mat2, i, axis=0)
            )
            result = op.MatMul(a, b)
            if a_is_2d and b_is_2d:
                result = op.Unsqueeze(result, [0])
            outputs.append(result)
            start = end

        if a_is_2d and b_is_2d:
            return op.Concat(*outputs, axis=0)
        axis = 0 if a_is_2d else 1
        result = op.Concat(*outputs, axis=axis)
        # PyTorch allocates the full output shape even if the last offset is short.
        # Its unwritten tail is unspecified; fill it with zeros rather than shrink it.
        length = (
            op.Shape(self, start=0, end=1)
            if a_is_2d
            else op.Shape(mat2, start=1, end=2)
        )
        pads = op.Concat(
            builder.constant([0], dtype=ir.DataType.INT64),
            op.Sub(length, end),
            axis=0,
        )
        return op.Pad(result, pads, None, [axis])

    res = op.MatMul(self, mat2)
    if bias is not None:
        res = op.Add(res, bias)
    if out_dtype is not None and out_dtype != -1:
        res = op.Cast(res, to=out_dtype)
    return res


@onnx_impl(aten.heaviside.default)
def aten_heaviside(
    op: TranslationContext, self: ir.Value, values: ir.Value
) -> ir.Value:
    """heaviside(Tensor self, Tensor values) -> Tensor"""

    zero = op.CastLike(0, self)
    one = op.CastLike(1, self)
    intermediate = op.Where(op.Less(self, zero), zero, one)

    return op.Where(op.Equal(self, zero), values, intermediate)


@onnx_impl(aten.histc.default)
def aten_histc(
    op: TranslationContext,
    self: ir.Value,
    bins: int = 100,
    min: float = 0.0,
    max: float = 0.0,
) -> ir.Value:
    """histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor"""
    if min == max:
        # This ONNXScript implementation precomputes static bin edges and cannot
        # faithfully reproduce torch.histc's dynamic behavior when min == max
        # (including the default min=0, max=0, which infers the range from data).
        raise NotImplementedError(
            f"aten_histc with min == max ({min}) is not supported in this export path."
        )
    delta = (max - min) / (bins * 1.0)
    values = [min + delta * i for i in range(bins + 1)]

    flat_self = op.Reshape(self, [-1])
    computation_type = self.dtype

    cond = op.And(
        op.GreaterOrEqual(flat_self, op.CastLike([min], self)),
        op.LessOrEqual(flat_self, op.CastLike([max], self)),
    )

    if not (self.type.dtype not in {ir.DataType.INT32, ir.DataType.INT64}):
        raise ValueError(f"torch.histc only works on float but {self.type.dtype=}")

    cond = op.And(cond, op.Not(op.IsNaN(flat_self)))
    # max is included.
    dtype = self.type.dtype.numpy()
    values = np.array(values, dtype=dtype)
    values[-1] = np.nextafter(values[-1], np.array(np.inf, dtype=dtype), dtype=dtype)
    typed_values = op.Constant(value=ir.tensor(values, dtype=self.type.dtype))

    clipped = op.Where(cond, flat_self, op.CastLike([min - 1], self))
    bins = op.Unsqueeze(typed_values, [1])

    less = op.Cast(
        op.Less(op.Unsqueeze(clipped, [0]), bins),
        to=computation_type,
    )
    sums = op.ReduceSum(less, [1], keepdims=0)
    res = op.Sub(
        op.Slice(sums, [1], op.Shape(sums), [0]),
        op.Slice(sums, [0], [-1], [0]),
    )
    return res


@onnx_impl(aten.is_nonzero.default)
def aten_is_nonzero(op: TranslationContext, self: ir.Value) -> ir.Value:
    """is_nonzero(Tensor self) -> bool"""
    builder = op.builder

    # if size != 1, return False
    # else [0],[True],[0.0] return True, others return False
    result = op.Equal(op.Size(self), 1)
    then_branch = ir.Graph([], [], nodes=[], name="nonzero_single")
    local = GraphBuilder(then_branch, opset=builder.opset)
    local_op = TranslationContext(local)
    scalar = local_op.Reshape(
        local_op.Cast(self, to=ir.DataType.BOOL),
        local.constant([], dtype=ir.DataType.INT64),
    )
    common_ops.finish_subgraph(op, then_branch, [scalar])
    else_branch = ir.Graph([], [], nodes=[], name="nonzero_multiple")
    local = GraphBuilder(else_branch, opset=builder.opset)
    local_op = TranslationContext(local)
    common_ops.finish_subgraph(
        op, else_branch, [local.constant(False, dtype=ir.DataType.BOOL)]
    )
    return builder.node(
        "If",
        [result],
        attributes={"then_branch": then_branch, "else_branch": else_branch},
        output_types=[ir.TensorType(ir.DataType.BOOL)],
    )[0]


@onnx_impl(aten.logcumsumexp.default)
def aten_logcumsumexp(op: TranslationContext, self: ir.Value, dim: int) -> ir.Value:
    """logcumsumexp(Tensor self, int dim) -> Tensor"""

    if len(self.shape) == 0:
        result = self
    else:
        # Make dim 1-d
        dims = op.Unsqueeze(dim, [0])
        # This uses the max trick to avoid overflow:
        # Assuming A = [a_1, a_2, ..., a_n] and the output
        # out = [out_1, out_2, ..., out_n], then
        # out_i = log(cumsum(exp(A)))_i
        #       = log(exp(a_1) + ... + exp(a_i))
        #       = log(exp(a_1) + ... + exp(a_i)) - max(A) + max(A)
        #       = log((exp(a_1) + ... + exp(a_i)) / exp(max(A))) + max(A)
        #       = log(exp(a_1-max(A)) + ... + exp(a_i-max(A))) + max(A)
        #       = log(sum<j=1...i>(exp(a_j - max(A)))) + max(A)
        # Vectorizing for all i, we get the expression below.
        self_max = op.ReduceMax(self, dims)
        result = op.Add(
            op.Log(
                op.CumSum(
                    op.Exp(op.Sub(self, self_max)),
                    dim,
                ),
            ),
            self_max,
        )
    return result


@onnx_impl(aten.logdet.default)
def aten_logdet(op: TranslationContext, self: ir.Value) -> ir.Value:
    """logdet(Tensor self) -> Tensor"""

    return op.Log(op.Det(self))


@onnx_impl(aten.native_dropout.default)
def aten_native_dropout(
    op: TranslationContext, input: ir.Value, p: float, train: bool = True
) -> tuple[ir.Value, ir.Value]:
    """native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)"""
    builder = op.builder
    if p == 1 and train is True:
        result = aten_dropout(op, input, p, train)
        mask = op.Expand(
            builder.constant(False, dtype=ir.DataType.BOOL),
            op.Shape(input),
        )
        return result, mask

    result, mask = op.Dropout(input, p, train, outputs=2)
    return result, mask


@onnx_impl(aten.pixel_shuffle.default)
def aten_pixel_shuffle(
    op: TranslationContext, self: ir.Value, upscale_factor: int
) -> ir.Value:
    """pixel_shuffle(Tensor self, int upscale_factor) -> Tensor"""
    builder = op.builder
    if len(self.shape) == 4:
        return op.DepthToSpace(self, blocksize=upscale_factor, mode="CRD")

    # Reshaping input by collapsing all leading dimensions to match ONNX op requirement (4D)
    batch_dims = op.Shape(self, end=-3)
    chw_in_dims = op.Shape(self, start=-3)

    reshaped_self = op.Reshape(
        self,
        op.Concat(
            builder.constant([-1], dtype=ir.DataType.INT64),
            chw_in_dims,
            axis=0,
        ),
    )
    depth_to_space = op.DepthToSpace(
        reshaped_self, blocksize=upscale_factor, mode="CRD"
    )
    final_dims = op.Shape(depth_to_space, start=1)
    output_shape = op.Concat(batch_dims, final_dims, axis=0)
    return op.Reshape(depth_to_space, output_shape, allowzero=True)


@onnx_impl(aten.pixel_unshuffle.default)
def aten_pixel_unshuffle(
    op: TranslationContext, self: ir.Value, downscale_factor: int
) -> ir.Value:
    """pixel_unshuffle(Tensor self, int downscale_factor) -> Tensor"""
    builder = op.builder
    # pixel_unshuffle is the inverse of pixel_shuffle (which uses DepthToSpace with mode="CRD").
    # SpaceToDepth only supports DCR channel ordering, so we implement via Reshape->Transpose->Reshape
    # to get the correct CRD ordering.
    # Input: [..., C, H*r, W*r] -> Output: [..., C*r*r, H, W]

    # Reshaping input by collapsing all leading dimensions to match ONNX op requirement (4D)
    batch_dims = op.Shape(self, end=-3)
    chw_in_dims = op.Shape(self, start=-3)
    self_4d = op.Reshape(
        self,
        op.Concat(
            builder.constant([-1], dtype=ir.DataType.INT64),
            chw_in_dims,
            axis=0,
        ),
    )

    r = builder.constant([downscale_factor], dtype=ir.DataType.INT64)
    c = op.Shape(self_4d, start=1, end=2)
    h_r = op.Shape(self_4d, start=2, end=3)
    w_r = op.Shape(self_4d, start=3, end=4)
    h = op.Div(h_r, r)
    w = op.Div(w_r, r)

    # Step 1: Reshape to [batch, C, H, r, W, r]
    shape_6d = op.Concat(
        builder.constant([-1], dtype=ir.DataType.INT64), c, h, r, w, r, axis=0
    )
    tmp = op.Reshape(self_4d, shape_6d)

    # Step 2: Transpose to [batch, C, r, r, H, W] (inverse of CRD DepthToSpace transpose)
    tmp = op.Transpose(tmp, perm=[0, 1, 3, 5, 2, 4])

    # Step 3: Reshape to [batch, C*r*r, H, W]
    c_out = op.Mul(c, op.Mul(r, r))
    shape_4d_out = op.Concat(
        builder.constant([-1], dtype=ir.DataType.INT64), c_out, h, w, axis=0
    )
    pixel_unshuffled = op.Reshape(tmp, shape_4d_out)

    final_dims = op.Shape(pixel_unshuffled, start=1)
    output_shape = op.Concat(batch_dims, final_dims, axis=0)
    return op.Reshape(pixel_unshuffled, output_shape, allowzero=True)


@onnx_impl(aten.polar.default)
def aten_polar(op: TranslationContext, abs: ir.Value, angle: ir.Value) -> ir.Value:
    """polar(Tensor abs, Tensor angle) -> Tensor"""

    real = op.Unsqueeze(op.Mul(abs, op.Cos(angle)), [-1])
    imag = op.Unsqueeze(op.Mul(abs, op.Sin(angle)), [-1])
    return op.Concat(real, imag, axis=-1)


@onnx_impl((aten.prelu.default, aten._prelu_kernel.default))
def aten_prelu(op: TranslationContext, self: ir.Value, weight: ir.Value) -> ir.Value:
    """prelu(Tensor self, Tensor weight) -> Tensor"""

    rank = len(self.shape)
    if rank == 0:
        # e.g. self: [], weight: [1]
        weight = op.Squeeze(weight)
    elif rank >= 2:
        # e.g. self: [5,10,5], weight: [10]
        weight = op.Reshape(weight, [1, -1] + [1] * (rank - 2))
    return op.PRelu(self, weight)


@onnx_impl(aten.roll.default)
def aten_roll(
    op: TranslationContext,
    self: ir.Value,
    shifts: Sequence[int],
    dims: Sequence[int] = (),
) -> ir.Value:
    """roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor"""

    if isinstance(shifts, int):
        shifts = [shifts]

    if isinstance(dims, int):
        dims = [dims]

    self_rank = len(self.shape)
    if self_rank == 0:
        return op.Identity(self)
    elif 0 in self.shape:
        # A tensor with no elements rolls to itself, which is what torch returns. It also
        # keeps a zero length out of the modulo in the helpers below, but only when the
        # length is known to be zero at export time. See the limitation noted there.
        return op.Identity(self)

    # NOTE: In pytorch, default value of dims is an empty list.
    if len(dims) == 0:  # Empty sequence
        if not (len(shifts) == 1):
            raise ValueError("shifts should be a single integer if dims is empty")
        return _aten_roll_shift_no_dim_onnx(op, self, shifts[0])
    else:
        if not (len(shifts) == len(dims)):
            raise ValueError(
                "Invalid metadata in aten_roll: len(shifts) == len(dims). Supply known tensor shapes and dtypes."
            )
        result = self
        for i, shift in enumerate(shifts):
            dim = dims[i]
            # PyTorch accepts negative dim as reversed counting
            if dim < 0:
                dim = self_rank + dim
            result = _aten_roll_shift_and_dim_onnx(op, result, shift, dim)
        return result


@onnx_impl(aten.roll.default, complex=True)
def aten_roll_complex(
    op: TranslationContext,
    self: ir.Value,
    shifts: Sequence[int],
    dims: Sequence[int] = (),
) -> ir.Value:
    """roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor"""

    if isinstance(shifts, int):
        shifts = [shifts]

    if isinstance(dims, int):
        dims = [dims]

    self_rank = len(self.shape)
    if self_rank == 1:
        return op.Identity(self)

    if 0 in self.shape:
        # Same as aten_roll: a tensor with no elements rolls to itself. The trailing
        # dimension that carries the real and imaginary parts is never zero, so this
        # only ever sees a dimension torch can see.
        return op.Identity(self)

    self_real = op.Slice(self, [0], [1], [-1])
    self_imag = op.Slice(self, [1], [2], [-1])
    if not dims:
        if not (len(shifts) == 1):
            raise ValueError("shifts should be a single integer if dims is empty")
        shift_real = _aten_roll_shift_no_dim_onnx(op, self_real, shifts[0])
        shift_imag = _aten_roll_shift_no_dim_onnx(op, self_imag, shifts[0])

        result = op.Concat(shift_real, shift_imag, axis=-1)

    else:
        if not (len(shifts) == len(dims)):
            raise ValueError(
                "Invalid metadata in aten_roll_complex: len(shifts) == len(dims). Supply known tensor shapes and dtypes."
            )
        for i, dim in enumerate(dims):
            if dim < 0:
                # Account for the complex dimension in ONNX
                dim = self_rank + dim - 1
            self_real = _aten_roll_shift_and_dim_onnx(op, self_real, shifts[i], dim)
            self_imag = _aten_roll_shift_and_dim_onnx(op, self_imag, shifts[i], dim)

        result = op.Concat(self_real, self_imag, axis=-1)
    return result


@onnx_impl(aten.selu.default)
def aten_selu(op: TranslationContext, self: ir.Value) -> ir.Value:
    """selu(Tensor self) -> Tensor"""

    return op.Selu(self)


@onnx_impl(aten.sigmoid.default)
def aten_sigmoid(op: TranslationContext, self: ir.Value) -> ir.Value:
    """sigmoid(Tensor self) -> Tensor"""

    return op.Sigmoid(self)


@onnx_impl(aten.signbit.default)
def aten_signbit(op: TranslationContext, self: ir.Value) -> ir.Value:
    """signbit(Tensor self) -> Tensor"""

    if self.dtype == ir.DataType.BOOL:
        return op.ConstantOfShape(op.Shape(self), value=ir.tensor([False]))

    # -0.0 should return True, but ONNX does not have an appropriate operator to handle it.
    return op.Less(
        self,
        op.Constant(value=ir.tensor([0], dtype=self.dtype)),
    )


@onnx_impl((aten.softmax.int, aten.special_softmax.default))
def aten_softmax(
    op: TranslationContext, self: ir.Value, dim: int, dtype: int = -1
) -> ir.Value:
    """softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor"""
    builder = op.builder

    self_is_scalar = len(self.shape) == 0
    if self_is_scalar:
        self = op.Unsqueeze(self, builder.constant([0], dtype=ir.DataType.INT64))
    result = op.Softmax(self, axis=dim)
    if dtype != -1 and dtype is not None:
        result = op.Cast(result, to=dtype)
    if self_is_scalar:
        # Convert to scalar when input is scalar
        result = op.Squeeze(result)

    return result


@onnx_impl(aten.stft.default)
def aten_stft(
    op: TranslationContext,
    self: ir.Value,
    n_fft: int,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: ir.Value | None = None,
    normalized: bool = False,
    onesided: bool | None = None,
    return_complex: bool | None = None,
) -> ir.Value:
    """stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool? onesided=None, bool? return_complex=None) -> Tensor"""
    builder = op.builder

    # NOTE: regardless of the value of return_complex, we always return a real representation.
    del return_complex

    # Get STFT sizes
    if hop_length is None:
        # core dump
        # hop_length = op.Div(op.Constant(value_ints=n_fft), op.Constant(value_ints=[4]))
        hop_length = n_fft // 4

    # ONNX's STFT requires a rank-3 signal of shape [batch_size, signal_length, 1]
    # (the trailing dimension is the real component). torch.stft accepts rank-1 or
    # rank-2 signals.
    is_signal_rank1 = len(self.shape) == 1
    if is_signal_rank1:
        # [signal_length] -> [1, signal_length, 1]: add batch dim and trailing real-component dim
        self = op.Unsqueeze(self, builder.constant([0, -1], dtype=ir.DataType.INT64))
    else:
        # [batch_size, signal_length] -> [batch_size, signal_length, 1]
        self = op.Unsqueeze(self, builder.constant([-1], dtype=ir.DataType.INT64))

    # Get window and make sure it's the same size as `win_length` or `n_fft`
    if window is not None:
        # first dimension
        n_win = op.Shape(window, start=0, end=1)
        # Center window around zeros if needed (required by ONNX's STFT)
        condition = op.Less(op.Squeeze(n_win), n_fft)
        then_branch = ir.Graph([], [], nodes=[], name="stft_pad_window")
        outer = builder
        builder = GraphBuilder(then_branch, opset=outer.opset)
        then_op = TranslationContext(builder)
        left = then_op.Div(
            then_op.Sub(n_fft, n_win),
            builder.constant([2], dtype=ir.DataType.INT64),
        )

        right = then_op.Sub(then_op.Sub(n_fft, left), n_win)
        left = then_op.Reshape(left, builder.constant([1], dtype=ir.DataType.INT64))
        right = then_op.Reshape(right, builder.constant([1], dtype=ir.DataType.INT64))

        left_win = then_op.Expand(builder.constant([0], dtype=ir.DataType.INT64), left)
        right_win = then_op.Expand(
            builder.constant([0], dtype=ir.DataType.INT64), right
        )
        right_win = then_op.CastLike(right_win, window)
        left_win = then_op.CastLike(left_win, window)
        padded_window = then_op.Concat(left_win, window, right_win, axis=0)
        common_ops.finish_subgraph(op, then_branch, [padded_window])
        else_branch = ir.Graph([], [], nodes=[], name="stft_keep_window")
        builder = GraphBuilder(else_branch, opset=outer.opset)
        else_op = TranslationContext(builder)
        common_ops.finish_subgraph(else_op, else_branch, [else_op.Identity(window)])
        builder = outer
        window = builder.node(
            "If",
            [condition],
            attributes={"then_branch": then_branch, "else_branch": else_branch},
            output_types=[window.type],
        )[0]
    elif window is None:
        if win_length is not None:
            window = _create_window_from_win_length(op, win_length, n_fft)
        else:
            window = _create_window_from_n_fft(op, n_fft)

    if onesided is None or onesided:
        onesided = 1
    else:
        onesided = 0
    window = op.CastLike(window, self)
    result = op.STFT(self, hop_length, window, n_fft, onesided=onesided)
    result = op.Transpose(result, perm=[0, 2, 1, 3])
    # Remove batch dimension, if needed
    if is_signal_rank1:
        result = op.Squeeze(result, builder.constant([0], dtype=ir.DataType.INT64))

    # Normalize, if needed
    if normalized:
        result = _normalize_fft_result(op, self, result, n_fft)

    return result


@onnx_impl(aten.tril.default)
def aten_tril(op: TranslationContext, self: ir.Value, diagonal: int = 0) -> ir.Value:
    """tril(Tensor self, int diagonal=0) -> Tensor"""

    return op.Trilu(self, diagonal, upper=0)


@onnx_impl(aten.triu.default)
def aten_triu(op: TranslationContext, self: ir.Value, diagonal: int = 0) -> ir.Value:
    """triu(Tensor self, int diagonal=0) -> Tensor"""

    return op.Trilu(self, diagonal, upper=1)


@onnx_impl(aten.unique_consecutive.default)
def aten_unique_consecutive(
    op: TranslationContext,
    x: ir.Value,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)"""
    builder = op.builder
    if x.dtype not in {ir.DataType.INT64, ir.DataType.INT32}:
        raise ValueError(
            "unique_consecutive not implemented for other type than int32, int64"
        )
    self_input = x
    rank_x = len(x.shape)

    zero64 = op.Constant(value=ir.tensor([0], dtype=ir.DataType.INT64))
    minus_one = op.Constant(value=ir.tensor([-1], dtype=ir.DataType.INT64))

    if dim is None:
        if rank_x != 1:
            x = op.Reshape(x, minus_one)
    else:
        if not (rank_x == 1 and dim == 0):
            raise ValueError(
                f"Not implemented for x={x!r} with rank={rank_x} and dim={dim}."
            )

    # Hopefully this will never be equal to the first value of the tensor x
    # ideally we could do differently but with a higher cost
    # Compare only adjacent elements; an explicit first mask avoids a sentinel collision.
    lag = op.Slice(x, zero64, minus_one, zero64)
    tail = op.Slice(x, [1], op.Shape(x))
    eq = op.Equal(tail, lag)
    first = op.Expand(
        builder.constant([True], dtype=ir.DataType.BOOL),
        op.Min(op.Shape(x), [1]),
    )
    diff = op.Concat(first, op.Not(eq), axis=0)
    res = op.Compress(x, diff, axis=0)

    zero_no_dim = builder.constant(0, dtype=ir.DataType.INT64)
    one_no_dim = builder.constant(1, dtype=ir.DataType.INT64)
    one = builder.constant([1], dtype=ir.DataType.INT64)

    inverse = op.Sub(
        op.CumSum(op.Cast(diff, to=ir.DataType.INT64), 0),
        one,
    )
    shape_x = op.Shape(x)
    indices = op.Range(zero_no_dim, op.Squeeze(shape_x), one_no_dim)
    points = op.Compress(indices, diff, axis=0)
    lagp = op.Concat(
        op.Slice(points, one, op.Shape(points), zero64),
        op.Expand(
            shape_x,
            op.Min(op.Shape(points), [1]),
        ),
        axis=0,
    )
    counts = op.Sub(lagp, points)
    if return_inverse:
        inverse = op.Reshape(inverse, op.Shape(self_input), allowzero=True)
    else:
        inverse = builder.constant([], dtype=ir.DataType.INT64)
    if not return_counts:
        counts = builder.constant([], dtype=ir.DataType.INT64)
    return res, inverse, counts


def _unique_empty_inverse(
    op: TranslationContext, input_numel, inverse_indices, input_size
):
    builder = op.builder
    then_branch = ir.Graph([], [], nodes=[], name="unique_empty_input")
    local = GraphBuilder(then_branch, opset=builder.opset)
    local_op = TranslationContext(local)
    common_ops.finish_subgraph(
        op,
        then_branch,
        [local_op.Reshape(inverse_indices, input_size, allowzero=True)],
    )
    else_branch = ir.Graph([], [], nodes=[], name="unique_no_inverse")
    local = GraphBuilder(else_branch, opset=builder.opset)
    local_op = TranslationContext(local)
    common_ops.finish_subgraph(
        op, else_branch, [local.constant([], dtype=ir.DataType.INT64)]
    )
    condition = op.Equal(input_numel, 0)
    return builder.node(
        "If",
        [condition],
        attributes={"then_branch": then_branch, "else_branch": else_branch},
        output_types=[ir.TensorType(ir.DataType.INT64)],
    )[0]


@onnx_impl(aten._unique.default)
def aten__unique(
    op: TranslationContext,
    self: ir.Value,
    sorted: bool = True,  # pylint: disable=unused-argument
    return_inverse: bool = False,
) -> tuple[ir.Value, ir.Value]:
    """_unique(Tensor self, bool sorted=True, bool return_inverse=False) -> (Tensor, Tensor)"""

    unique_values, _, inverse_indices, _ = op.Unique(
        self, axis=None, sorted=True, outputs=4
    )
    input_size = op.Shape(self)
    if return_inverse:
        inverse_indices = op.Reshape(inverse_indices, input_size, allowzero=True)
    else:
        input_numel = op.ReduceProd(input_size, keepdims=False)
        inverse_indices = _unique_empty_inverse(
            op, input_numel, inverse_indices, input_size
        )
    return unique_values, inverse_indices


@onnx_impl(aten._unique2.default)
def aten__unique2(
    op: TranslationContext,
    self: ir.Value,
    sorted: bool = True,  # pylint: disable=unused-argument
    return_inverse: bool = False,
    return_counts: bool = False,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """_unique2(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)"""

    unique_values, _, inverse_indices, counts = op.Unique(
        self, axis=None, sorted=True, outputs=4
    )
    input_size = op.Shape(self)
    if return_inverse:
        inverse_indices = op.Reshape(inverse_indices, input_size, allowzero=True)
    else:
        input_numel = op.ReduceProd(input_size, keepdims=False)
        inverse_indices = _unique_empty_inverse(
            op, input_numel, inverse_indices, input_size
        )
    if not return_counts:
        counts = op.ConstantOfShape([0])
        counts = op.Cast(counts, to=ir.DataType.INT64)
    return unique_values, inverse_indices, counts


@onnx_impl(aten.unique_dim.default)
def aten_unique_dim(
    op: TranslationContext,
    self: ir.Value,
    dim: int,
    sorted: bool = True,  # pylint: disable=unused-argument
    return_inverse: bool = False,
    return_counts: bool = False,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)"""
    builder = op.builder

    unique_values, _, inverse_indices, counts = op.Unique(
        self, axis=dim, sorted=True, outputs=4
    )
    input_size = op.Shape(self)
    # Normalize dim to be non-negative
    input_ndim = op.Max(
        op.Size(input_size),
        builder.constant([1], dtype=ir.DataType.INT64),
    )
    dim = op.Mod(dim, input_ndim)
    if return_inverse:
        inverse_indices = op.Reshape(
            inverse_indices,
            op.Reshape(
                op.Slice(input_size, dim, op.Add(dim, 1)),
                builder.constant([-1], dtype=ir.DataType.INT64),
            ),
        )
    else:
        inverse_indices = op.ConstantOfShape([0])
        inverse_indices = op.Cast(inverse_indices, to=ir.DataType.INT64)
    if return_counts:
        output_size = op.Shape(unique_values)
        counts = op.Reshape(
            counts,
            op.Reshape(
                op.Slice(output_size, dim, op.Add(dim, 1)),
                [-1],
            ),
        )
    else:
        counts = op.ConstantOfShape([0])
        counts = op.Cast(counts, to=ir.DataType.INT64)
    return unique_values, inverse_indices, counts


def _aten_fake_quantize_per_tensor_affine(
    op: TranslationContext,
    self: ir.Value,
    scale: float | ir.Value,
    zero_point: int | ir.Value,
    quant_min: int,
    quant_max: int,
) -> ir.Value:
    # NOTE: (0, 127) is allowed as special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    if (quant_min, quant_max) not in [(0, 255), (-128, 127), (0, 127)]:
        raise NotImplementedError(
            "For (quant_min, quant_max), ONNX allows only "
            "(0, 127), (0, 255) and (-128, 127). "
            f"Got ({quant_min}, {quant_max})",
        )

    if quant_min == 0:
        int_dtype = ir.DataType.UINT8
    else:
        int_dtype = ir.DataType.INT8

    # TODO: When opset >= 19, remove this cast
    orig_dtype = self.type.dtype
    if self.type.dtype not in {ir.DataType.FLOAT, ir.DataType.INT32}:
        self = op.Cast(self, to=ir.DataType.FLOAT)

    # TODO: When opset >= 19, relex the condition for this cast
    if isinstance(scale, float) or scale.type.dtype != ir.DataType.FLOAT:
        scale = op.Cast(scale, to=ir.DataType.FLOAT)

    if isinstance(zero_point, int) or zero_point.type.dtype != int_dtype:
        zero_point = op.Cast(zero_point, to=int_dtype)

    quantized = op.QuantizeLinear(self, scale, zero_point)

    # See comment about, PyTorch-specific (0, 127) handling
    if (quant_min, quant_max) == (0, 127):
        const_127 = op.Cast(127, to=int_dtype)
        quantized = op.Clip(quantized, None, const_127)

    output = op.DequantizeLinear(quantized, scale, zero_point)

    # TODO: When opset >= 23, remove this cast and set output_dtype on DequantizeLinear
    if orig_dtype != ir.DataType.FLOAT:
        output = op.Cast(output, to=orig_dtype)

    return output


def _aten_roll_shift_no_dim_onnx(
    op: TranslationContext, self: ir.Value, shift: int
) -> ir.Value:
    builder = op.builder
    # Source implementation context (the native divisor is guarded below):
    # The element count is the divisor of the Mod below, and Mod by zero is undefined in
    # ONNX. Both callers return a tensor with no elements unchanged before reaching here.
    # Known limitation: those early returns and this assert only cover a length known to be
    # zero at export time. A dynamic dimension that is zero at run time still reaches Mod
    # with a zero divisor. Depending on the runtime, its version and the platform, that can
    # raise an error, crash the process, or quietly return a value.
    if not (self.shape is None or 0 not in self.shape):
        raise ValueError(
            "the element count must not be zero because Mod by zero is undefined"
        )

    neg_1 = builder.constant([-1], dtype=ir.DataType.INT64)
    # flatten the self tensor: from [[A,B],[C,D]] to [A,B,C,D]
    self_flatten = op.Reshape(self, neg_1)
    total_length = op.Shape(self_flatten)
    divisor = op.Max(total_length, builder.constant([1], dtype=ir.DataType.INT64))
    # Compute slice length. roll is circular, so the shift is taken modulo the number
    # of elements. For [A,B,C,D], if shift is 1, slice_length = 3, means move [A,B,C]
    # to the end. The effect equals to move [D] to the beginning.
    slice_length = op.Mod(builder.constant([-shift], dtype=ir.DataType.INT64), divisor)
    # Get second part of the tensor, e.g. [A,B,C]
    suffix = op.Slice(
        self_flatten,
        builder.constant([0], dtype=ir.DataType.INT64),
        slice_length,
    )
    # Get first part of the tensor, e.g. [D]
    prefix = op.Slice(self_flatten, slice_length, total_length)
    # Concat first+second together, e.g. [D,A,B,C]
    result = op.Concat(prefix, suffix, axis=0)
    # allowzero so a dimension that is only zero at run time stays zero here, rather than
    # being read as "copy the input dimension" against a flattened tensor.
    return op.Reshape(result, op.Shape(self), allowzero=True)


def _aten_roll_shift_and_dim_onnx(
    op: TranslationContext, self: ir.Value, shift: int, dim: int
) -> ir.Value:
    builder = op.builder
    # Source implementation context (the native divisor is guarded below):
    # dim must already be normalized to a nonnegative axis, because Shape below
    # reads an empty range when start is negative and end is zero.
    # The length of that dimension is the divisor of the Mod below, and Mod by zero is
    # undefined in ONNX. Both callers return a tensor with no elements unchanged before
    # reaching here.
    # Known limitation: as in _aten_roll_shift_no_dim_onnx, if that dimension is dynamic and
    # is zero at run time, Mod still gets a zero divisor.
    if not (self.shape is None or self.shape[dim] != 0):
        raise ValueError(
            "the dimension length must not be zero because Mod by zero is undefined"
        )

    dim_tensor = builder.constant([dim], dtype=ir.DataType.INT64)
    dim_length = op.Shape(self, start=dim, end=dim + 1)
    divisor = op.Max(dim_length, builder.constant([1], dtype=ir.DataType.INT64))
    # roll is circular, so the shift is taken modulo the length of the dimension
    slice_length = op.Mod(builder.constant([-shift], dtype=ir.DataType.INT64), divisor)
    # from [A,B,C,D] -> [D,A,B,C], [D] is prefix, [A,B,C] is suffix
    suffix = op.Slice(
        self,
        builder.constant([0], dtype=ir.DataType.INT64),
        slice_length,
        dim_tensor,
    )
    prefix = op.Slice(self, slice_length, dim_length, dim_tensor)
    result = op.Concat(prefix, suffix, axis=dim)
    return result


def _create_window_from_win_length(
    op: TranslationContext, win_length: int, n_fft: int
) -> ir.Value:
    builder = op.builder
    left = op.Div(
        op.Sub(n_fft, win_length),
        builder.constant([2], dtype=ir.DataType.INT64),
    )

    right = op.Sub(op.Sub(n_fft, left), win_length)
    left = op.Reshape(left, builder.constant([1], dtype=ir.DataType.INT64))
    right = op.Reshape(right, builder.constant([1], dtype=ir.DataType.INT64))
    win_length = op.Reshape(win_length, builder.constant([1], dtype=ir.DataType.INT64))

    left_win = op.Expand(builder.constant([0], dtype=ir.DataType.INT64), left)
    right_win = op.Expand(builder.constant([0], dtype=ir.DataType.INT64), right)
    window_list = op.Expand(builder.constant([1], dtype=ir.DataType.INT64), win_length)
    return op.Concat(left_win, window_list, right_win, axis=0)


def _create_window_from_n_fft(op: TranslationContext, n_fft: int) -> ir.Value:
    builder = op.builder
    n_fft_tensor = op.Reshape(n_fft, builder.constant([1], dtype=ir.DataType.INT64))
    window = op.Expand(builder.constant([1], dtype=ir.DataType.INT64), n_fft_tensor)
    return window


def _normalize_fft_result(
    op: TranslationContext, signal: ir.Value, result: ir.Value, n_fft: int
) -> ir.Value:
    builder = op.builder
    n_fft_tensor = op.Reshape(n_fft, builder.constant([1], dtype=ir.DataType.INT64))
    sqrt_nfft = op.Sqrt(op.CastLike(n_fft_tensor, signal))
    result = op.Div(result, sqrt_nfft)
    return result
