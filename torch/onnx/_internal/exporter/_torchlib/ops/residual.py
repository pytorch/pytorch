# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""PyTorch-owned native ONNX IR residual translations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir

import torch
from torch.onnx._internal.exporter._ir_builder import current_builder, GraphBuilder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx._internal.exporter._torchlib.ops import common as common_ops, core


if TYPE_CHECKING:
    from collections.abc import Sequence


aten = torch.ops.aten


@onnx_impl(aten._log_softmax.default)
def aten__log_softmax(self: ir.Value, dim: int, half_to_float: bool) -> ir.Value:
    """_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"""
    builder = current_builder()

    self_is_scalar = len(self.shape) == 0
    if half_to_float and self.dtype in {ir.DataType.FLOAT16, ir.DataType.BFLOAT16}:
        self = builder.call("Cast", self, to=ir.DataType.FLOAT)[0]
    if self_is_scalar:
        self = builder.call(
            "Unsqueeze", self, builder.constant([0], dtype=ir.DataType.INT64)
        )[0]
    result = builder.call("LogSoftmax", self, axis=dim)[0]
    if self_is_scalar:
        result = builder.call(
            "Squeeze", result, builder.constant([0], dtype=ir.DataType.INT64)
        )[0]
    return result


@onnx_impl(aten._softmax.default)
def aten__softmax(self: ir.Value, dim: int, half_to_float: bool) -> ir.Value:
    """_softmax(Tensor self, int dim, bool half_to_float) -> Tensor"""
    builder = current_builder()

    self_is_scalar = len(self.shape) == 0

    if half_to_float and self.dtype in {ir.DataType.FLOAT16, ir.DataType.BFLOAT16}:
        self = builder.call("Cast", self, to=ir.DataType.FLOAT)[0]

    if self_is_scalar:
        self = builder.call(
            "Unsqueeze", self, builder.constant([0], dtype=ir.DataType.INT64)
        )[0]
    result = builder.call("Softmax", self, axis=dim)[0]
    if self_is_scalar:
        # Convert to scalar when input is scalar
        result = builder.call("Squeeze", result)[0]

    return result


@onnx_impl(aten.allclose.default)
def aten_allclose(
    self: ir.Value,
    other: ir.Value,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> ir.Value:
    """allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool"""
    builder = current_builder()

    # FIXME: check equal_nan when self and other are all NaN
    # |input - other| <= atol + rtol x |other|
    left_part = builder.call("Abs", builder.call("Sub", self, other)[0])[0]
    right_part = builder.call(
        "Add", atol, builder.call("Mul", rtol, builder.call("Abs", other)[0])[0]
    )[0]
    is_close = builder.call("LessOrEqual", left_part, right_part)[0]
    is_close_int = builder.call("Cast", is_close, to=ir.DataType.INT8)[0]

    # If min is 0, some elements are not close -> allclose is False
    # If min is 1, all elements are close -> allclose is True
    return builder.call(
        "Cast",
        builder.call("ReduceMin", is_close_int, keepdims=False)[0],
        to=ir.DataType.BOOL,
    )[0]


@onnx_impl(aten.bilinear.default)
def aten_bilinear(
    input1: ir.Value,
    input2: ir.Value,
    weight: ir.Value,
    bias: ir.Value | None = None,
) -> ir.Value:
    """bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias=None) -> Tensor"""
    builder = current_builder()

    # Bilinear transformation: y = x1^T A x2 + b
    # input1 shape: (..., in1_features)
    # input2 shape: (..., in2_features)
    # weight shape: (out_features, in1_features, in2_features)
    # bias shape: (out_features) - optional
    # output shape: (..., out_features)

    # input1 and input2 must have identical batch dimensions
    # Use MatMul to compute the bilinear transformation
    batch_size = builder.call("Shape", input1, start=0, end=-1)[0]
    input1_shape = builder.call("Shape", input1, start=-1)[0]
    input2_shape = builder.call("Shape", input2, start=-1)[0]
    output_shape = builder.call("Shape", weight, start=0, end=1)[0]
    neg_1 = builder.constant([-1], dtype=ir.DataType.INT64)

    # (out_features, in1_features, in2_features) -> (in1_features, out_features, in2_features)
    W_permute = builder.call("Transpose", weight, perm=[1, 0, 2])[0]

    # (in1_features, out_features, in2_features) -> (in1_features, out_features * in2_features)
    W_flat = builder.call(
        "Reshape",
        W_permute,
        builder.call(
            "Concat",
            input1_shape,
            builder.call("Mul", output_shape, input2_shape)[0],
            axis=0,
        )[0],
    )[0]

    # (..., in1_features) @ (in1_features, out_features * in2_features) -> (..., out_features * in2_features)
    tmp = builder.call("MatMul", input1, W_flat)[0]

    # (..., out_features * in2_features) -> (..., out_features, in2_features)
    tmp = builder.call(
        "Reshape",
        tmp,
        builder.call("Concat", batch_size, output_shape, input2_shape, axis=0)[0],
    )[0]

    # (..., in2_features) -> (..., in2_features, 1)
    #   -> (..., out_features, in2_features) @ (..., in2_features, 1)
    #   -> (..., out_features, 1) -> (..., out_features)
    result = builder.call(
        "Squeeze",
        builder.call("MatMul", tmp, builder.call("Unsqueeze", input2, neg_1)[0])[0],
        neg_1,
    )[0]

    if bias is not None:
        result = builder.call("Add", result, bias)[0]

    return result


@onnx_impl(aten.bincount.default)
def aten_bincount(
    self: ir.Value, weights: ir.Value | None = None, minlength: int = 0
) -> ir.Value:
    """bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor

    ``weights`` is not supported. Negative inputs are rejected by torch and are not
    handled here (ONNX integer ops would wrap them around).
    """
    builder = current_builder()
    if weights is not None:
        raise NotImplementedError("aten::bincount with weights is not supported.")

    self = builder.call("Cast", self, to=ir.DataType.INT64)[0]
    axis_0 = builder.constant([0], dtype=ir.DataType.INT64)
    # Append a 0 so ReduceMax is defined even when ``self`` is empty. It only sizes the
    # output and never contributes to the counts (the scatter below uses ``self``).
    data_max = builder.call(
        "Unsqueeze",
        builder.call(
            "ReduceMax",
            builder.call(
                "Concat", self, builder.constant([0], dtype=ir.DataType.INT64), axis=0
            )[0],
            keepdims=0,
        )[0],
        axis_0,
    )[0]
    # An empty input yields depth 0, so the output is empty unless ``minlength`` applies.
    non_empty = builder.call(
        "Unsqueeze",
        builder.call(
            "Cast",
            builder.call(
                "Greater",
                builder.call("Size", self)[0],
                builder.constant(0, dtype=ir.DataType.INT64),
            )[0],
            to=ir.DataType.INT64,
        )[0],
        axis_0,
    )[0]
    depth = builder.call(
        "Mul",
        builder.call("Add", data_max, builder.constant([1], dtype=ir.DataType.INT64))[
            0
        ],
        non_empty,
    )[0]
    if minlength > 0:
        depth = builder.call(
            "Max", depth, builder.constant([minlength], dtype=ir.DataType.INT64)
        )[0]

    # Scatter-add 1 for each value into a zero vector of length ``depth``. This uses
    # O(N + depth) memory instead of the dense O(N * depth) one-hot, and behaves
    # correctly for empty inputs.
    zeros = builder.call("Expand", builder.constant(0, dtype=ir.DataType.INT64), depth)[
        0
    ]
    ones = builder.call(
        "Expand",
        builder.constant(1, dtype=ir.DataType.INT64),
        builder.call("Shape", self)[0],
    )[0]
    return builder.call("ScatterElements", zeros, self, ones, axis=0, reduction="add")[
        0
    ]


@onnx_impl(aten.complex.default)
def aten_complex(real: ir.Value, imag: ir.Value) -> ir.Value:
    """complex(Tensor real, Tensor imag) -> Tensor"""
    builder = current_builder()

    # Broadcast the real and imaginary parts to the same shape
    broadcasted_shape = core._shape_of_broadcast_tensors(real, imag)
    real = builder.call("Expand", real, broadcasted_shape)[0]
    imag = builder.call("Expand", imag, broadcasted_shape)[0]

    return builder.call(
        "Concat",
        builder.call("Unsqueeze", real, [-1])[0],
        builder.call("Unsqueeze", imag, [-1])[0],
        axis=-1,
    )[0]


@onnx_impl(aten.constant_pad_nd.default)
def aten_constant_pad_nd(
    self: ir.Value, pad: Sequence[ir.Value], value: float = 0.0
) -> ir.Value:
    """constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor"""
    builder = current_builder()

    # The desired order of paddings is
    # dim_0_begin, dim_1_begin, ... , dim_0_end, ..., dim_n_end.
    # n is the dimension of input.
    # assume zero-dimensions in the beginning
    rank = len(self.shape)
    paddings = list(pad) + [0] * (rank * 2 - len(pad))
    # reverse order and collate first beginnings and then ends
    paddings = paddings[-2::-2] + paddings[-1::-2]
    constant_value = builder.call("Constant", value=ir.tensor(value, dtype=self.dtype))[
        0
    ]

    return builder.call("Pad", self, common_ops.merge_dims(paddings), constant_value)[0]


@onnx_impl((aten.cross.default, aten.linalg_cross.default))
def aten_cross(self: ir.Value, other: ir.Value, dim: int = -1) -> ir.Value:
    """cross(Tensor self, Tensor other, int? dim=None) -> Tensor"""
    builder = current_builder()

    # Reference https://en.wikipedia.org/w/index.php?title=Cross_product&oldid=1143125073
    a1, a2, a3 = builder.call("Split", self, axis=dim, num_outputs=3, _num_outputs=3)
    b1, b2, b3 = builder.call("Split", other, axis=dim, num_outputs=3, _num_outputs=3)
    # Broadcasting is implicitly supported by Mul
    c1 = builder.call(
        "Sub", builder.call("Mul", a2, b3)[0], builder.call("Mul", a3, b2)[0]
    )[0]
    c2 = builder.call(
        "Sub", builder.call("Mul", a3, b1)[0], builder.call("Mul", a1, b3)[0]
    )[0]
    c3 = builder.call(
        "Sub", builder.call("Mul", a1, b2)[0], builder.call("Mul", a2, b1)[0]
    )[0]

    return builder.call("Concat", c1, c2, c3, axis=dim)[0]


@onnx_impl(aten.dropout.default)
def aten_dropout(
    input: ir.Value, p: float | ir.Value, train: bool | ir.Value
) -> ir.Value:
    """dropout(Tensor input, float p, bool train) -> Tensor"""
    builder = current_builder()
    # ONNX Dropout requires ratio < 1; PyTorch also supports dropping every element.
    if isinstance(p, (int, float)) and p == 1 and train is True:
        return builder.call("Mul", input, builder.call("CastLike", 0, input)[0])[0]

    if input.shape is not None and len(input.shape) == 0:
        input = builder.call(
            "Reshape", input, builder.constant([-1], dtype=ir.DataType.INT64)
        )[0]
        result, _ = builder.call("Dropout", input, p, train, _num_outputs=2)
        result = builder.call("Squeeze", result)[0]
    else:
        result, _ = builder.call("Dropout", input, p, train, _num_outputs=2)

    return result


@onnx_impl(aten.equal.default)
def aten_equal(self: ir.Value, other: ir.Value) -> ir.Value:
    """equal(Tensor self, Tensor other) -> bool"""
    builder = current_builder()

    # NOTE: Torch aten::equal returns a single Boolean while ONNX Equal is elementwise.
    # The equivalent Torch op with ONNX Equal is aten::eq.
    elementwise_equal = builder.call("Equal", self, other)[0]
    elementwise_equal_int = builder.call(
        "Cast", elementwise_equal, to=ir.DataType.INT64
    )[0]
    # ReduceMin does not support bool. So we cast to int64
    all_equal = builder.call("ReduceMin", elementwise_equal_int, keepdims=False)[0]
    return builder.call("Cast", all_equal, to=ir.DataType.BOOL)[0]


@onnx_impl(aten.fake_quantize_per_channel_affine.default)
def aten_fake_quantize_per_channel_affine(
    self: ir.Value,
    scale: ir.Value,  # float32 specifically!
    zero_point: ir.Value,  # int32, float32 or float16 only!
    axis: int,
    quant_min: int,
    quant_max: int,
) -> ir.Value:
    """fake_quantize_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> Tensor"""
    builder = current_builder()

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
        self = builder.call("Cast", self, to=ir.DataType.FLOAT)[0]

    if zero_point.type.dtype == ir.DataType.INT32:
        zero_point = builder.call("Cast", zero_point, to=int_dtype)[0]
    else:
        raise NotImplementedError(
            "ONNX only supports integer values for the zero_point parameter. "
            f"Got {zero_point.type.dtype}",
        )

    quantized = builder.call("QuantizeLinear", self, scale, zero_point, axis=axis)[0]

    # See comment about, PyTorch-specific (0, 127) handling
    if (quant_min, quant_max) == (0, 127):
        const_127 = builder.call("Cast", 127, to=int_dtype)[0]
        quantized = builder.call("Clip", quantized, None, const_127)[0]

    output = builder.call("DequantizeLinear", quantized, scale, zero_point, axis=axis)[
        0
    ]

    # TODO: When opset >= 23, remove this cast and set output_dtype on DequantizeLinear
    if orig_dtype != ir.DataType.FLOAT:
        output = builder.call("Cast", output, to=orig_dtype)[0]

    return output


@onnx_impl(aten.fake_quantize_per_tensor_affine.default)
def aten_fake_quantize_per_tensor_affine(
    self: ir.Value,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
) -> ir.Value:
    """fake_quantize_per_tensor_affine(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> Tensor"""

    return _aten_fake_quantize_per_tensor_affine(
        self, scale, zero_point, quant_min, quant_max
    )


@onnx_impl(aten.fake_quantize_per_tensor_affine.tensor_qparams)
def aten_fake_quantize_per_tensor_affine_tensor_qparams(
    self: ir.Value,
    scale: ir.Value,
    zero_point: ir.Value,
    quant_min: int,
    quant_max: int,
) -> ir.Value:
    """fake_quantize_per_tensor_affine(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max) -> Tensor"""

    return _aten_fake_quantize_per_tensor_affine(
        self, scale, zero_point, quant_min, quant_max
    )


@onnx_impl(aten.grid_sampler.default)
def aten_grid_sampler(
    input: ir.Value,
    grid: ir.Value,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> ir.Value:
    """grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor"""
    builder = current_builder()

    inter_mode_options = (
        ("linear", "nearest", "cubic")
        if builder.opset >= 20
        else ("bilinear", "nearest", "bicubic")
    )
    inter_mode_str = inter_mode_options[interpolation_mode]

    padding_mode_options = ("zeros", "border", "reflection")
    padding_mode_str = padding_mode_options[padding_mode]

    return builder.call(
        "GridSample",
        input,
        grid,
        align_corners=align_corners,
        mode=inter_mode_str,
        padding_mode=padding_mode_str,
    )[0]


@onnx_impl(aten.grid_sampler_2d.default)
def aten_grid_sampler_2d(
    input: ir.Value,
    grid: ir.Value,
    interpolation_mode: int,
    padding_mode: int,
    align_corners: bool,
) -> ir.Value:
    """grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor"""
    builder = current_builder()

    inter_mode_options = (
        ("linear", "nearest", "cubic")
        if builder.opset >= 20
        else ("bilinear", "nearest", "bicubic")
    )
    inter_mode_str = inter_mode_options[interpolation_mode]

    padding_mode_options = ("zeros", "border", "reflection")
    padding_mode_str = padding_mode_options[padding_mode]

    return builder.call(
        "GridSample",
        input,
        grid,
        align_corners=align_corners,
        mode=inter_mode_str,
        padding_mode=padding_mode_str,
    )[0]


@onnx_impl(aten._grouped_mm.default)
def aten_grouped_mm(
    self: ir.Value,
    mat2: ir.Value,
    offs: ir.Value | None = None,
    bias: ir.Value | None = None,
    out_dtype: int = -1,
) -> ir.Value:
    """_grouped_mm(Tensor self, Tensor mat2, *, Tensor? offs=None, Tensor? bias=None, int? out_dtype=None) -> Tensor"""
    builder = current_builder()

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
            output_shape = builder.call(
                "Concat",
                builder.call("Shape", self, start=a_rank - 2, end=a_rank - 1)[0],
                builder.call("Shape", mat2, start=b_rank - 1, end=b_rank)[0],
                axis=0,
            )[0]
            if a_is_2d and b_is_2d:
                output_shape = builder.call(
                    "Concat",
                    builder.constant([0], dtype=ir.DataType.INT64),
                    output_shape,
                    axis=0,
                )[0]
            return builder.call(
                "Expand", builder.call("CastLike", 0, self)[0], output_shape
            )[0]

        # Only the group count is static. Boundaries remain values in the graph.
        ends = builder.call("Cast", offs, to=ir.DataType.INT64)[0]
        start = builder.constant([0], dtype=ir.DataType.INT64)
        outputs = []
        for i in range(groups):
            end = builder.call("Gather", ends, [i], axis=0)[0]
            a = (
                builder.call("Slice", self, start, end, [1 if b_is_2d else 0])[0]
                if a_is_2d
                else builder.call("Gather", self, i, axis=0)[0]
            )
            b = (
                builder.call("Slice", mat2, start, end, [0 if a_is_2d else 1])[0]
                if b_is_2d
                else builder.call("Gather", mat2, i, axis=0)[0]
            )
            result = builder.call("MatMul", a, b)[0]
            if a_is_2d and b_is_2d:
                result = builder.call("Unsqueeze", result, [0])[0]
            outputs.append(result)
            start = end

        if a_is_2d and b_is_2d:
            return builder.call("Concat", *outputs, axis=0)[0]
        axis = 0 if a_is_2d else 1
        result = builder.call("Concat", *outputs, axis=axis)[0]
        # PyTorch allocates the full output shape even if the last offset is short.
        # Its unwritten tail is unspecified; fill it with zeros rather than shrink it.
        length = (
            builder.call("Shape", self, start=0, end=1)[0]
            if a_is_2d
            else builder.call("Shape", mat2, start=1, end=2)[0]
        )
        pads = builder.call(
            "Concat",
            builder.constant([0], dtype=ir.DataType.INT64),
            builder.call("Sub", length, end)[0],
            axis=0,
        )[0]
        return builder.call("Pad", result, pads, None, [axis])[0]

    res = builder.call("MatMul", self, mat2)[0]
    if bias is not None:
        res = builder.call("Add", res, bias)[0]
    if out_dtype is not None and out_dtype != -1:
        res = builder.call("Cast", res, to=out_dtype)[0]
    return res


@onnx_impl(aten.heaviside.default)
def aten_heaviside(self: ir.Value, values: ir.Value) -> ir.Value:
    """heaviside(Tensor self, Tensor values) -> Tensor"""
    builder = current_builder()

    zero = builder.call("CastLike", 0, self)[0]
    one = builder.call("CastLike", 1, self)[0]
    intermediate = builder.call(
        "Where", builder.call("Less", self, zero)[0], zero, one
    )[0]

    return builder.call(
        "Where", builder.call("Equal", self, zero)[0], values, intermediate
    )[0]


@onnx_impl(aten.histc.default)
def aten_histc(
    self: ir.Value, bins: int = 100, min: float = 0.0, max: float = 0.0
) -> ir.Value:
    """histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor"""
    builder = current_builder()
    if min == max:
        # This ONNXScript implementation precomputes static bin edges and cannot
        # faithfully reproduce torch.histc's dynamic behavior when min == max
        # (including the default min=0, max=0, which infers the range from data).
        raise NotImplementedError(
            f"aten_histc with min == max ({min}) is not supported in this export path."
        )
    delta = (max - min) / (bins * 1.0)
    values = [min + delta * i for i in range(bins + 1)]

    flat_self = builder.call("Reshape", self, [-1])[0]
    computation_type = self.dtype

    cond = builder.call(
        "And",
        builder.call(
            "GreaterOrEqual", flat_self, builder.call("CastLike", [min], self)[0]
        )[0],
        builder.call(
            "LessOrEqual", flat_self, builder.call("CastLike", [max], self)[0]
        )[0],
    )[0]

    if not (self.type.dtype not in {ir.DataType.INT32, ir.DataType.INT64}):
        raise ValueError(f"torch.histc only works on float but {self.type.dtype=}")

    cond = builder.call(
        "And", cond, builder.call("Not", builder.call("IsNaN", flat_self)[0])[0]
    )[0]
    # max is included.
    dtype = self.type.dtype.numpy()
    values = np.array(values, dtype=dtype)
    values[-1] = np.nextafter(values[-1], np.array(np.inf, dtype=dtype), dtype=dtype)
    typed_values = builder.call(
        "Constant", value=ir.tensor(values, dtype=self.type.dtype)
    )[0]

    clipped = builder.call(
        "Where", cond, flat_self, builder.call("CastLike", [min - 1], self)[0]
    )[0]
    bins = builder.call("Unsqueeze", typed_values, [1])[0]

    less = builder.call(
        "Cast",
        builder.call("Less", builder.call("Unsqueeze", clipped, [0])[0], bins)[0],
        to=computation_type,
    )[0]
    sums = builder.call("ReduceSum", less, [1], keepdims=0)[0]
    res = builder.call(
        "Sub",
        builder.call("Slice", sums, [1], builder.call("Shape", sums)[0], [0])[0],
        builder.call("Slice", sums, [0], [-1], [0])[0],
    )[0]
    return res


@onnx_impl(aten.is_nonzero.default)
def aten_is_nonzero(self: ir.Value) -> ir.Value:
    """is_nonzero(Tensor self) -> bool"""
    builder = current_builder()

    # if size != 1, return False
    # else [0],[True],[0.0] return True, others return False
    result = builder.call("Equal", builder.call("Size", self)[0], 1)[0]
    then_branch = ir.Graph([], [], nodes=[], name="nonzero_single")
    local = GraphBuilder(then_branch, opset=builder.opset)
    scalar = local.call(
        "Reshape",
        local.call("Cast", self, to=ir.DataType.BOOL)[0],
        local.constant([], dtype=ir.DataType.INT64),
    )[0]
    common_ops.finish_subgraph(then_branch, [scalar])
    else_branch = ir.Graph([], [], nodes=[], name="nonzero_multiple")
    local = GraphBuilder(else_branch, opset=builder.opset)
    common_ops.finish_subgraph(
        else_branch, [local.constant(False, dtype=ir.DataType.BOOL)]
    )
    return builder.node(
        "If",
        [result],
        attributes={"then_branch": then_branch, "else_branch": else_branch},
        output_types=[ir.TensorType(ir.DataType.BOOL)],
    )[0]


@onnx_impl(aten.logcumsumexp.default)
def aten_logcumsumexp(self: ir.Value, dim: int) -> ir.Value:
    """logcumsumexp(Tensor self, int dim) -> Tensor"""
    builder = current_builder()

    if len(self.shape) == 0:
        result = self
    else:
        # Make dim 1-d
        dims = builder.call("Unsqueeze", dim, [0])[0]
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
        self_max = builder.call("ReduceMax", self, dims)[0]
        result = builder.call(
            "Add",
            builder.call(
                "Log",
                builder.call(
                    "CumSum",
                    builder.call("Exp", builder.call("Sub", self, self_max)[0])[0],
                    dim,
                )[0],
            )[0],
            self_max,
        )[0]
    return result


@onnx_impl(aten.logdet.default)
def aten_logdet(self: ir.Value) -> ir.Value:
    """logdet(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Log", builder.call("Det", self)[0])[0]


@onnx_impl(aten.native_dropout.default)
def aten_native_dropout(
    input: ir.Value, p: float, train: bool = True
) -> tuple[ir.Value, ir.Value]:
    """native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)"""
    builder = current_builder()
    if p == 1 and train is True:
        result = aten_dropout(input, p, train)
        mask = builder.call(
            "Expand",
            builder.constant(False, dtype=ir.DataType.BOOL),
            builder.call("Shape", input)[0],
        )[0]
        return result, mask

    result, mask = builder.call("Dropout", input, p, train, _num_outputs=2)
    return result, mask


@onnx_impl(aten.pixel_shuffle.default)
def aten_pixel_shuffle(self: ir.Value, upscale_factor: int) -> ir.Value:
    """pixel_shuffle(Tensor self, int upscale_factor) -> Tensor"""
    builder = current_builder()
    if len(self.shape) == 4:
        return builder.call("DepthToSpace", self, blocksize=upscale_factor, mode="CRD")[
            0
        ]

    # Reshaping input by collapsing all leading dimensions to match ONNX op requirement (4D)
    batch_dims = builder.call("Shape", self, end=-3)[0]
    chw_in_dims = builder.call("Shape", self, start=-3)[0]

    reshaped_self = builder.call(
        "Reshape",
        self,
        builder.call(
            "Concat",
            builder.constant([-1], dtype=ir.DataType.INT64),
            chw_in_dims,
            axis=0,
        )[0],
    )[0]
    depth_to_space = builder.call(
        "DepthToSpace", reshaped_self, blocksize=upscale_factor, mode="CRD"
    )[0]
    final_dims = builder.call("Shape", depth_to_space, start=1)[0]
    output_shape = builder.call("Concat", batch_dims, final_dims, axis=0)[0]
    return builder.call("Reshape", depth_to_space, output_shape, allowzero=True)[0]


@onnx_impl(aten.pixel_unshuffle.default)
def aten_pixel_unshuffle(self: ir.Value, downscale_factor: int) -> ir.Value:
    """pixel_unshuffle(Tensor self, int downscale_factor) -> Tensor"""
    builder = current_builder()
    # pixel_unshuffle is the inverse of pixel_shuffle (which uses DepthToSpace with mode="CRD").
    # SpaceToDepth only supports DCR channel ordering, so we implement via Reshape->Transpose->Reshape
    # to get the correct CRD ordering.
    # Input: [..., C, H*r, W*r] -> Output: [..., C*r*r, H, W]

    # Reshaping input by collapsing all leading dimensions to match ONNX op requirement (4D)
    batch_dims = builder.call("Shape", self, end=-3)[0]
    chw_in_dims = builder.call("Shape", self, start=-3)[0]
    self_4d = builder.call(
        "Reshape",
        self,
        builder.call(
            "Concat",
            builder.constant([-1], dtype=ir.DataType.INT64),
            chw_in_dims,
            axis=0,
        )[0],
    )[0]

    r = builder.constant([downscale_factor], dtype=ir.DataType.INT64)
    c = builder.call("Shape", self_4d, start=1, end=2)[0]
    h_r = builder.call("Shape", self_4d, start=2, end=3)[0]
    w_r = builder.call("Shape", self_4d, start=3, end=4)[0]
    h = builder.call("Div", h_r, r)[0]
    w = builder.call("Div", w_r, r)[0]

    # Step 1: Reshape to [batch, C, H, r, W, r]
    shape_6d = builder.call(
        "Concat", builder.constant([-1], dtype=ir.DataType.INT64), c, h, r, w, r, axis=0
    )[0]
    tmp = builder.call("Reshape", self_4d, shape_6d)[0]

    # Step 2: Transpose to [batch, C, r, r, H, W] (inverse of CRD DepthToSpace transpose)
    tmp = builder.call("Transpose", tmp, perm=[0, 1, 3, 5, 2, 4])[0]

    # Step 3: Reshape to [batch, C*r*r, H, W]
    c_out = builder.call("Mul", c, builder.call("Mul", r, r)[0])[0]
    shape_4d_out = builder.call(
        "Concat", builder.constant([-1], dtype=ir.DataType.INT64), c_out, h, w, axis=0
    )[0]
    pixel_unshuffled = builder.call("Reshape", tmp, shape_4d_out)[0]

    final_dims = builder.call("Shape", pixel_unshuffled, start=1)[0]
    output_shape = builder.call("Concat", batch_dims, final_dims, axis=0)[0]
    return builder.call("Reshape", pixel_unshuffled, output_shape, allowzero=True)[0]


@onnx_impl(aten.polar.default)
def aten_polar(abs: ir.Value, angle: ir.Value) -> ir.Value:
    """polar(Tensor abs, Tensor angle) -> Tensor"""
    builder = current_builder()

    real = builder.call(
        "Unsqueeze", builder.call("Mul", abs, builder.call("Cos", angle)[0])[0], [-1]
    )[0]
    imag = builder.call(
        "Unsqueeze", builder.call("Mul", abs, builder.call("Sin", angle)[0])[0], [-1]
    )[0]
    return builder.call("Concat", real, imag, axis=-1)[0]


@onnx_impl((aten.prelu.default, aten._prelu_kernel.default))
def aten_prelu(self: ir.Value, weight: ir.Value) -> ir.Value:
    """prelu(Tensor self, Tensor weight) -> Tensor"""
    builder = current_builder()

    rank = len(self.shape)
    if rank == 0:
        # e.g. self: [], weight: [1]
        weight = builder.call("Squeeze", weight)[0]
    elif rank >= 2:
        # e.g. self: [5,10,5], weight: [10]
        weight = builder.call("Reshape", weight, [1, -1] + [1] * (rank - 2))[0]
    return builder.call("PRelu", self, weight)[0]


@onnx_impl(aten.roll.default)
def aten_roll(
    self: ir.Value, shifts: Sequence[int], dims: Sequence[int] = ()
) -> ir.Value:
    """roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor"""
    builder = current_builder()

    if isinstance(shifts, int):
        shifts = [shifts]

    if isinstance(dims, int):
        dims = [dims]

    self_rank = len(self.shape)
    if self_rank == 0:
        return builder.call("Identity", self)[0]
    elif 0 in self.shape:
        # A tensor with no elements rolls to itself, which is what torch returns. It also
        # keeps a zero length out of the modulo in the helpers below, but only when the
        # length is known to be zero at export time. See the limitation noted there.
        return builder.call("Identity", self)[0]

    # NOTE: In pytorch, default value of dims is an empty list.
    if len(dims) == 0:  # Empty sequence
        if not (len(shifts) == 1):
            raise ValueError("shifts should be a single integer if dims is empty")
        return _aten_roll_shift_no_dim_onnx(self, shifts[0])
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
            result = _aten_roll_shift_and_dim_onnx(result, shift, dim)
        return result


@onnx_impl(aten.roll.default, complex=True)
def aten_roll_complex(
    self: ir.Value, shifts: Sequence[int], dims: Sequence[int] = ()
) -> ir.Value:
    """roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor"""
    builder = current_builder()

    if isinstance(shifts, int):
        shifts = [shifts]

    if isinstance(dims, int):
        dims = [dims]

    self_rank = len(self.shape)
    if self_rank == 1:
        return builder.call("Identity", self)[0]

    if 0 in self.shape:
        # Same as aten_roll: a tensor with no elements rolls to itself. The trailing
        # dimension that carries the real and imaginary parts is never zero, so this
        # only ever sees a dimension torch can see.
        return builder.call("Identity", self)[0]

    self_real = builder.call("Slice", self, [0], [1], [-1])[0]
    self_imag = builder.call("Slice", self, [1], [2], [-1])[0]
    if not dims:
        if not (len(shifts) == 1):
            raise ValueError("shifts should be a single integer if dims is empty")
        shift_real = _aten_roll_shift_no_dim_onnx(self_real, shifts[0])
        shift_imag = _aten_roll_shift_no_dim_onnx(self_imag, shifts[0])

        result = builder.call("Concat", shift_real, shift_imag, axis=-1)[0]

    else:
        if not (len(shifts) == len(dims)):
            raise ValueError(
                "Invalid metadata in aten_roll_complex: len(shifts) == len(dims). Supply known tensor shapes and dtypes."
            )
        for i, dim in enumerate(dims):
            if dim < 0:
                # Account for the complex dimension in ONNX
                dim = self_rank + dim - 1
            self_real = _aten_roll_shift_and_dim_onnx(self_real, shifts[i], dim)
            self_imag = _aten_roll_shift_and_dim_onnx(self_imag, shifts[i], dim)

        result = builder.call("Concat", self_real, self_imag, axis=-1)[0]
    return result


@onnx_impl(aten.selu.default)
def aten_selu(self: ir.Value) -> ir.Value:
    """selu(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Selu", self)[0]


@onnx_impl(aten.sigmoid.default)
def aten_sigmoid(self: ir.Value) -> ir.Value:
    """sigmoid(Tensor self) -> Tensor"""
    builder = current_builder()

    return builder.call("Sigmoid", self)[0]


@onnx_impl(aten.signbit.default)
def aten_signbit(self: ir.Value) -> ir.Value:
    """signbit(Tensor self) -> Tensor"""
    builder = current_builder()

    if self.dtype == ir.DataType.BOOL:
        return builder.call(
            "ConstantOfShape", builder.call("Shape", self)[0], value=ir.tensor([False])
        )[0]

    # -0.0 should return True, but ONNX does not have an appropriate operator to handle it.
    return builder.call(
        "Less",
        self,
        builder.call("Constant", value=ir.tensor([0], dtype=self.dtype))[0],
    )[0]


@onnx_impl((aten.softmax.int, aten.special_softmax.default))
def aten_softmax(self: ir.Value, dim: int, dtype: int = -1) -> ir.Value:
    """softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor"""
    builder = current_builder()

    self_is_scalar = len(self.shape) == 0
    if self_is_scalar:
        self = builder.call(
            "Unsqueeze", self, builder.constant([0], dtype=ir.DataType.INT64)
        )[0]
    result = builder.call("Softmax", self, axis=dim)[0]
    if dtype != -1 and dtype is not None:
        result = builder.call("Cast", result, to=dtype)[0]
    if self_is_scalar:
        # Convert to scalar when input is scalar
        result = builder.call("Squeeze", result)[0]

    return result


@onnx_impl(aten.stft.default)
def aten_stft(
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
    builder = current_builder()

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
        self = builder.call(
            "Unsqueeze", self, builder.constant([0, -1], dtype=ir.DataType.INT64)
        )[0]
    else:
        # [batch_size, signal_length] -> [batch_size, signal_length, 1]
        self = builder.call(
            "Unsqueeze", self, builder.constant([-1], dtype=ir.DataType.INT64)
        )[0]

    # Get window and make sure it's the same size as `win_length` or `n_fft`
    if window is not None:
        # first dimension
        n_win = builder.call("Shape", window, start=0, end=1)[0]
        # Center window around zeros if needed (required by ONNX's STFT)
        condition = builder.call("Less", builder.call("Squeeze", n_win)[0], n_fft)[0]
        then_branch = ir.Graph([], [], nodes=[], name="stft_pad_window")
        outer = builder
        builder = GraphBuilder(then_branch, opset=outer.opset)
        with builder.activate():
            left = builder.call(
                "Div",
                builder.call("Sub", n_fft, n_win)[0],
                builder.constant([2], dtype=ir.DataType.INT64),
            )[0]

            right = builder.call("Sub", builder.call("Sub", n_fft, left)[0], n_win)[0]
            left = builder.call(
                "Reshape", left, builder.constant([1], dtype=ir.DataType.INT64)
            )[0]
            right = builder.call(
                "Reshape", right, builder.constant([1], dtype=ir.DataType.INT64)
            )[0]

            left_win = builder.call(
                "Expand", builder.constant([0], dtype=ir.DataType.INT64), left
            )[0]
            right_win = builder.call(
                "Expand", builder.constant([0], dtype=ir.DataType.INT64), right
            )[0]
            right_win = builder.call("CastLike", right_win, window)[0]
            left_win = builder.call("CastLike", left_win, window)[0]
            padded_window = builder.call("Concat", left_win, window, right_win, axis=0)[
                0
            ]
        common_ops.finish_subgraph(then_branch, [padded_window])
        else_branch = ir.Graph([], [], nodes=[], name="stft_keep_window")
        builder = GraphBuilder(else_branch, opset=outer.opset)
        common_ops.finish_subgraph(else_branch, [builder.call("Identity", window)[0]])
        builder = outer
        window = builder.node(
            "If",
            [condition],
            attributes={"then_branch": then_branch, "else_branch": else_branch},
            output_types=[window.type],
        )[0]
    elif window is None:
        if win_length is not None:
            window = _create_window_from_win_length(win_length, n_fft)
        else:
            window = _create_window_from_n_fft(n_fft)

    if onesided is None or onesided:
        onesided = 1
    else:
        onesided = 0
    window = builder.call("CastLike", window, self)[0]
    result = builder.call("STFT", self, hop_length, window, n_fft, onesided=onesided)[0]
    result = builder.call("Transpose", result, perm=[0, 2, 1, 3])[0]
    # Remove batch dimension, if needed
    if is_signal_rank1:
        result = builder.call(
            "Squeeze", result, builder.constant([0], dtype=ir.DataType.INT64)
        )[0]

    # Normalize, if needed
    if normalized:
        result = _normalize_fft_result(self, result, n_fft)

    return result


@onnx_impl(aten.tril.default)
def aten_tril(self: ir.Value, diagonal: int = 0) -> ir.Value:
    """tril(Tensor self, int diagonal=0) -> Tensor"""
    builder = current_builder()

    return builder.call("Trilu", self, diagonal, upper=0)[0]


@onnx_impl(aten.triu.default)
def aten_triu(self: ir.Value, diagonal: int = 0) -> ir.Value:
    """triu(Tensor self, int diagonal=0) -> Tensor"""
    builder = current_builder()

    return builder.call("Trilu", self, diagonal, upper=1)[0]


@onnx_impl(aten.unique_consecutive.default)
def aten_unique_consecutive(
    x: ir.Value,
    return_inverse: bool = False,
    return_counts: bool = False,
    dim: int | None = None,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)"""
    builder = current_builder()
    if x.dtype not in {ir.DataType.INT64, ir.DataType.INT32}:
        raise ValueError(
            "unique_consecutive not implemented for other type than int32, int64"
        )
    self_input = x
    rank_x = len(x.shape)

    zero64 = builder.call("Constant", value=ir.tensor([0], dtype=ir.DataType.INT64))[0]
    minus_one = builder.call(
        "Constant", value=ir.tensor([-1], dtype=ir.DataType.INT64)
    )[0]

    if dim is None:
        if rank_x != 1:
            x = builder.call("Reshape", x, minus_one)[0]
    else:
        if not (rank_x == 1 and dim == 0):
            raise ValueError(
                f"Not implemented for x={x!r} with rank={rank_x} and dim={dim}."
            )

    # Hopefully this will never be equal to the first value of the tensor x
    # ideally we could do differently but with a higher cost
    # Compare only adjacent elements; an explicit first mask avoids a sentinel collision.
    lag = builder.call("Slice", x, zero64, minus_one, zero64)[0]
    tail = builder.call("Slice", x, [1], builder.call("Shape", x)[0])[0]
    eq = builder.call("Equal", tail, lag)[0]
    first = builder.call(
        "Expand",
        builder.constant([True], dtype=ir.DataType.BOOL),
        builder.call("Min", builder.call("Shape", x)[0], [1])[0],
    )[0]
    diff = builder.call("Concat", first, builder.call("Not", eq)[0], axis=0)[0]
    res = builder.call("Compress", x, diff, axis=0)[0]

    zero_no_dim = builder.constant(0, dtype=ir.DataType.INT64)
    one_no_dim = builder.constant(1, dtype=ir.DataType.INT64)
    one = builder.constant([1], dtype=ir.DataType.INT64)

    inverse = builder.call(
        "Sub",
        builder.call("CumSum", builder.call("Cast", diff, to=ir.DataType.INT64)[0], 0)[
            0
        ],
        one,
    )[0]
    shape_x = builder.call("Shape", x)[0]
    indices = builder.call(
        "Range", zero_no_dim, builder.call("Squeeze", shape_x)[0], one_no_dim
    )[0]
    points = builder.call("Compress", indices, diff, axis=0)[0]
    lagp = builder.call(
        "Concat",
        builder.call("Slice", points, one, builder.call("Shape", points)[0], zero64)[0],
        builder.call(
            "Expand",
            shape_x,
            builder.call("Min", builder.call("Shape", points)[0], [1])[0],
        )[0],
        axis=0,
    )[0]
    counts = builder.call("Sub", lagp, points)[0]
    if return_inverse:
        inverse = builder.call(
            "Reshape", inverse, builder.call("Shape", self_input)[0], allowzero=True
        )[0]
    else:
        inverse = builder.constant([], dtype=ir.DataType.INT64)
    if not return_counts:
        counts = builder.constant([], dtype=ir.DataType.INT64)
    return res, inverse, counts


def _unique_empty_inverse(input_numel, inverse_indices, input_size):
    builder = current_builder()
    then_branch = ir.Graph([], [], nodes=[], name="unique_empty_input")
    local = GraphBuilder(then_branch, opset=builder.opset)
    common_ops.finish_subgraph(
        then_branch,
        [local.call("Reshape", inverse_indices, input_size, allowzero=True)[0]],
    )
    else_branch = ir.Graph([], [], nodes=[], name="unique_no_inverse")
    local = GraphBuilder(else_branch, opset=builder.opset)
    common_ops.finish_subgraph(
        else_branch, [local.constant([], dtype=ir.DataType.INT64)]
    )
    condition = builder.call("Equal", input_numel, 0)[0]
    return builder.node(
        "If",
        [condition],
        attributes={"then_branch": then_branch, "else_branch": else_branch},
        output_types=[ir.TensorType(ir.DataType.INT64)],
    )[0]


@onnx_impl(aten._unique.default)
def aten__unique(
    self: ir.Value,
    sorted: bool = True,  # pylint: disable=unused-argument
    return_inverse: bool = False,
) -> tuple[ir.Value, ir.Value]:
    """_unique(Tensor self, bool sorted=True, bool return_inverse=False) -> (Tensor, Tensor)"""
    builder = current_builder()

    unique_values, _, inverse_indices, _ = builder.call(
        "Unique", self, axis=None, sorted=True, _num_outputs=4
    )
    input_size = builder.call("Shape", self)[0]
    if return_inverse:
        inverse_indices = builder.call(
            "Reshape", inverse_indices, input_size, allowzero=True
        )[0]
    else:
        input_numel = builder.call("ReduceProd", input_size, keepdims=False)[0]
        inverse_indices = _unique_empty_inverse(
            input_numel, inverse_indices, input_size
        )
    return unique_values, inverse_indices


@onnx_impl(aten._unique2.default)
def aten__unique2(
    self: ir.Value,
    sorted: bool = True,  # pylint: disable=unused-argument
    return_inverse: bool = False,
    return_counts: bool = False,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """_unique2(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)"""
    builder = current_builder()

    unique_values, _, inverse_indices, counts = builder.call(
        "Unique", self, axis=None, sorted=True, _num_outputs=4
    )
    input_size = builder.call("Shape", self)[0]
    if return_inverse:
        inverse_indices = builder.call(
            "Reshape", inverse_indices, input_size, allowzero=True
        )[0]
    else:
        input_numel = builder.call("ReduceProd", input_size, keepdims=False)[0]
        inverse_indices = _unique_empty_inverse(
            input_numel, inverse_indices, input_size
        )
    if not return_counts:
        counts = builder.call("ConstantOfShape", [0])[0]
        counts = builder.call("Cast", counts, to=ir.DataType.INT64)[0]
    return unique_values, inverse_indices, counts


@onnx_impl(aten.unique_dim.default)
def aten_unique_dim(
    self: ir.Value,
    dim: int,
    sorted: bool = True,  # pylint: disable=unused-argument
    return_inverse: bool = False,
    return_counts: bool = False,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)"""
    builder = current_builder()

    unique_values, _, inverse_indices, counts = builder.call(
        "Unique", self, axis=dim, sorted=True, _num_outputs=4
    )
    input_size = builder.call("Shape", self)[0]
    # Normalize dim to be non-negative
    input_ndim = builder.call(
        "Max",
        builder.call("Size", input_size)[0],
        builder.constant([1], dtype=ir.DataType.INT64),
    )[0]
    dim = builder.call("Mod", dim, input_ndim)[0]
    if return_inverse:
        inverse_indices = builder.call(
            "Reshape",
            inverse_indices,
            builder.call(
                "Reshape",
                builder.call("Slice", input_size, dim, builder.call("Add", dim, 1)[0])[
                    0
                ],
                builder.constant([-1], dtype=ir.DataType.INT64),
            )[0],
        )[0]
    else:
        inverse_indices = builder.call("ConstantOfShape", [0])[0]
        inverse_indices = builder.call("Cast", inverse_indices, to=ir.DataType.INT64)[0]
    if return_counts:
        output_size = builder.call("Shape", unique_values)[0]
        counts = builder.call(
            "Reshape",
            counts,
            builder.call(
                "Reshape",
                builder.call("Slice", output_size, dim, builder.call("Add", dim, 1)[0])[
                    0
                ],
                [-1],
            )[0],
        )[0]
    else:
        counts = builder.call("ConstantOfShape", [0])[0]
        counts = builder.call("Cast", counts, to=ir.DataType.INT64)[0]
    return unique_values, inverse_indices, counts


def _aten_fake_quantize_per_tensor_affine(
    self: ir.Value,
    scale: float | ir.Value,
    zero_point: int | ir.Value,
    quant_min: int,
    quant_max: int,
) -> ir.Value:
    builder = current_builder()
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
        self = builder.call("Cast", self, to=ir.DataType.FLOAT)[0]

    # TODO: When opset >= 19, relex the condition for this cast
    if isinstance(scale, float) or scale.type.dtype != ir.DataType.FLOAT:
        scale = builder.call("Cast", scale, to=ir.DataType.FLOAT)[0]

    if isinstance(zero_point, int) or zero_point.type.dtype != int_dtype:
        zero_point = builder.call("Cast", zero_point, to=int_dtype)[0]

    quantized = builder.call("QuantizeLinear", self, scale, zero_point)[0]

    # See comment about, PyTorch-specific (0, 127) handling
    if (quant_min, quant_max) == (0, 127):
        const_127 = builder.call("Cast", 127, to=int_dtype)[0]
        quantized = builder.call("Clip", quantized, None, const_127)[0]

    output = builder.call("DequantizeLinear", quantized, scale, zero_point)[0]

    # TODO: When opset >= 23, remove this cast and set output_dtype on DequantizeLinear
    if orig_dtype != ir.DataType.FLOAT:
        output = builder.call("Cast", output, to=orig_dtype)[0]

    return output


def _aten_roll_shift_no_dim_onnx(self: ir.Value, shift: int) -> ir.Value:
    builder = current_builder()
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
    self_flatten = builder.call("Reshape", self, neg_1)[0]
    total_length = builder.call("Shape", self_flatten)[0]
    divisor = builder.call(
        "Max", total_length, builder.constant([1], dtype=ir.DataType.INT64)
    )[0]
    # Compute slice length. roll is circular, so the shift is taken modulo the number
    # of elements. For [A,B,C,D], if shift is 1, slice_length = 3, means move [A,B,C]
    # to the end. The effect equals to move [D] to the beginning.
    slice_length = builder.call(
        "Mod", builder.constant([-shift], dtype=ir.DataType.INT64), divisor
    )[0]
    # Get second part of the tensor, e.g. [A,B,C]
    suffix = builder.call(
        "Slice",
        self_flatten,
        builder.constant([0], dtype=ir.DataType.INT64),
        slice_length,
    )[0]
    # Get first part of the tensor, e.g. [D]
    prefix = builder.call("Slice", self_flatten, slice_length, total_length)[0]
    # Concat first+second together, e.g. [D,A,B,C]
    result = builder.call("Concat", prefix, suffix, axis=0)[0]
    # allowzero so a dimension that is only zero at run time stays zero here, rather than
    # being read as "copy the input dimension" against a flattened tensor.
    return builder.call(
        "Reshape", result, builder.call("Shape", self)[0], allowzero=True
    )[0]


def _aten_roll_shift_and_dim_onnx(self: ir.Value, shift: int, dim: int) -> ir.Value:
    builder = current_builder()
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
    dim_length = builder.call("Shape", self, start=dim, end=dim + 1)[0]
    divisor = builder.call(
        "Max", dim_length, builder.constant([1], dtype=ir.DataType.INT64)
    )[0]
    # roll is circular, so the shift is taken modulo the length of the dimension
    slice_length = builder.call(
        "Mod", builder.constant([-shift], dtype=ir.DataType.INT64), divisor
    )[0]
    # from [A,B,C,D] -> [D,A,B,C], [D] is prefix, [A,B,C] is suffix
    suffix = builder.call(
        "Slice",
        self,
        builder.constant([0], dtype=ir.DataType.INT64),
        slice_length,
        dim_tensor,
    )[0]
    prefix = builder.call("Slice", self, slice_length, dim_length, dim_tensor)[0]
    result = builder.call("Concat", prefix, suffix, axis=dim)[0]
    return result


def _create_window_from_win_length(win_length: int, n_fft: int) -> ir.Value:
    builder = current_builder()
    left = builder.call(
        "Div",
        builder.call("Sub", n_fft, win_length)[0],
        builder.constant([2], dtype=ir.DataType.INT64),
    )[0]

    right = builder.call("Sub", builder.call("Sub", n_fft, left)[0], win_length)[0]
    left = builder.call(
        "Reshape", left, builder.constant([1], dtype=ir.DataType.INT64)
    )[0]
    right = builder.call(
        "Reshape", right, builder.constant([1], dtype=ir.DataType.INT64)
    )[0]
    win_length = builder.call(
        "Reshape", win_length, builder.constant([1], dtype=ir.DataType.INT64)
    )[0]

    left_win = builder.call(
        "Expand", builder.constant([0], dtype=ir.DataType.INT64), left
    )[0]
    right_win = builder.call(
        "Expand", builder.constant([0], dtype=ir.DataType.INT64), right
    )[0]
    window_list = builder.call(
        "Expand", builder.constant([1], dtype=ir.DataType.INT64), win_length
    )[0]
    return builder.call("Concat", left_win, window_list, right_win, axis=0)[0]


def _create_window_from_n_fft(n_fft: int) -> ir.Value:
    builder = current_builder()
    n_fft_tensor = builder.call(
        "Reshape", n_fft, builder.constant([1], dtype=ir.DataType.INT64)
    )[0]
    window = builder.call(
        "Expand", builder.constant([1], dtype=ir.DataType.INT64), n_fft_tensor
    )[0]
    return window


def _normalize_fft_result(signal: ir.Value, result: ir.Value, n_fft: int) -> ir.Value:
    builder = current_builder()
    n_fft_tensor = builder.call(
        "Reshape", n_fft, builder.constant([1], dtype=ir.DataType.INT64)
    )[0]
    sqrt_nfft = builder.call("Sqrt", builder.call("CastLike", n_fft_tensor, signal)[0])[
        0
    ]
    result = builder.call("Div", result, sqrt_nfft)[0]
    return result
