"""torch.ops.quantized_decomposed operators."""

# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value"
# pyrefly: ignore-errors

from __future__ import annotations

import onnx_ir as ir

import torch
import torch.ao.quantization.fx._decomposed  # registers the quantized_decomposed ops
from torch.onnx._internal.exporter._ir_builder import current_builder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl
from torch.onnx._internal.exporter._torchlib.ops import common


quantized_decomposed = torch.ops.quantized_decomposed

# Full quantization range per dtype, from torch/ao/quantization/fx/_decomposed.py.
# int32 is the one dtype QuantizeLinear cannot emit, so it lowers to primitive ops.
_FULL_QUANTIZATION_RANGE = {
    ir.DataType.INT8: (-128, 127),
    ir.DataType.UINT8: (0, 255),
    ir.DataType.INT16: (-(2**15), 2**15 - 1),
    ir.DataType.UINT16: (0, 2**16 - 1),
    ir.DataType.INT32: (-(2**31), 2**31 - 1),
    ir.DataType.FLOAT8E4M3FN: (-448, 448),
    ir.DataType.FLOAT8E5M2: (-57344, 57344),
}

# Clip rejects float8, so narrowed float8 ranges keep QuantizeLinear's saturation.
_FLOAT8_DTYPES = frozenset(
    (
        ir.DataType.FLOAT8E4M3FN,
        ir.DataType.FLOAT8E4M3FNUZ,
        ir.DataType.FLOAT8E5M2,
        ir.DataType.FLOAT8E5M2FNUZ,
    )
)


def _clip(quantized, quant_min, quant_max):
    builder = current_builder()
    return builder.call(
        "Clip",
        quantized,
        builder.call("CastLike", quant_min, quantized)[0],
        builder.call("CastLike", quant_max, quantized)[0],
    )[0]


@onnx_impl(
    (
        quantized_decomposed.quantize_per_tensor.default,
        quantized_decomposed.quantize_per_tensor.tensor,
        quantized_decomposed.quantize_per_tensor.tensor2,
    ),
)
def quantized_decomposed_quantize_per_tensor(
    input: ir.Value,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: int,
) -> ir.Value:
    """quantize_per_tensor(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype) -> Tensor"""
    builder = current_builder()

    if dtype == ir.DataType.INT32:
        # Reference: clamp(round(input * (1 / scale)) + zero_point, quant_min, quant_max).to(int32).
        # The reciprocal divides in float64; float16/bfloat16 inputs compute in float32.
        if input.dtype in (ir.DataType.FLOAT16, ir.DataType.BFLOAT16):
            input = builder.call("Cast", input, to=ir.DataType.FLOAT)[0]
        if isinstance(scale, float):
            inv_scale = builder.call(
                "Constant", value=ir.tensor(1.0 / scale, dtype=ir.DataType.DOUBLE)
            )[0]
        else:
            inv_scale = builder.call(
                "Reciprocal",
                builder.call("Cast", scale, to=ir.DataType.DOUBLE)[0],
            )[0]
        quantized = builder.call(
            "Add",
            builder.call(
                "Round",
                builder.call(
                    "Mul",
                    input,
                    builder.call("CastLike", inv_scale, input)[0],
                )[0],
            )[0],
            builder.call("CastLike", zero_point, input)[0],
        )[0]
        # Cast float-to-int is undefined out of range, so clamping is required
        # even at the full int32 range.
        return builder.call("Cast", _clip(quantized, quant_min, quant_max), to=dtype)[0]

    quantized = builder.call(
        "QuantizeLinear",
        input,
        builder.call("CastLike", scale, input)[0],
        builder.call("Cast", zero_point, to=dtype)[0],
    )[0]
    if (
        dtype not in _FLOAT8_DTYPES
        and (quant_min, quant_max) != _FULL_QUANTIZATION_RANGE[dtype]
    ):
        # QuantizeLinear saturates to the full dtype range; narrower ranges need a Clip.
        quantized = _clip(quantized, quant_min, quant_max)
    return quantized


@onnx_impl(
    (
        quantized_decomposed.quantize_per_channel.default,
        "quantized_decomposed::quantize_per_channel.tensor",
        "quantized_decomposed::quantize_per_channel.tensor2",
    )
)
def quantized_decomposed_quantize_per_channel(
    input: ir.Value,
    scales: ir.Value,
    zero_points: ir.Value,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: int,
) -> ir.Value:
    """Affine per channel quantization for the Tensor using the same quantization
    parameters for each channel/axis to map from floating point to quantized values.

    Reference:
    https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/fx/_decomposed.py
    ``res = clamp(round(input / scales) + zero_points, quant_min, quant_max)``
    """
    builder = current_builder()

    if dtype == ir.DataType.INT32:
        shape = input.shape
        if shape is None:
            raise NotImplementedError(
                "Exporting int32 quantize_per_channel requires a static input "
                "rank to broadcast scales and zero_points along 'axis'"
            )
        rank = len(shape)
        if not -rank <= axis < rank:
            raise ValueError(
                f"quantize_per_channel expects axis in [{-rank}, {rank - 1}], got {axis}"
            )
        if axis < 0:
            axis += rank
        # The reciprocal is computed at the scales' own precision (float16/bfloat16
        # divide in float32 and round back); the multiply runs in the type promoted
        # from input and scales.
        compute_dtype = (
            ir.DataType.DOUBLE
            if scales.dtype == ir.DataType.DOUBLE
            else ir.DataType.FLOAT
        )
        if scales.dtype in (ir.DataType.FLOAT, ir.DataType.DOUBLE):
            reciprocal = builder.call("Reciprocal", scales)[0]
        else:
            reciprocal = builder.call(
                "Reciprocal",
                builder.call("Cast", scales, to=ir.DataType.FLOAT)[0],
            )[0]
            if scales.dtype in (ir.DataType.FLOAT16, ir.DataType.BFLOAT16):
                reciprocal = builder.call("Cast", reciprocal, to=scales.dtype)[0]
        if input.dtype != compute_dtype:
            input = builder.call("Cast", input, to=compute_dtype)[0]
        # Broadcast the 1-D values along 'axis'.
        axes = [i for i in range(rank) if i != axis]
        inv_scales = builder.call(
            "Unsqueeze", builder.call("CastLike", reciprocal, input)[0], axes
        )[0]
        zero_points = builder.call(
            "Unsqueeze", builder.call("CastLike", zero_points, input)[0], axes
        )[0]
        quantized = builder.call(
            "Add",
            builder.call("Round", builder.call("Mul", input, inv_scales)[0])[0],
            zero_points,
        )[0]
        # Cast float-to-int is undefined out of range, so clamping is required
        # even at the full int32 range.
        return builder.call("Cast", _clip(quantized, quant_min, quant_max), to=dtype)[0]

    # ONNX QuantizeLinear requires the scale to share the input element type and the
    # zero_point to share the (quantized) output element type. PyTorch passes scales as
    # float64 and zero_points as int64, so cast both to the expected ONNX types.
    quantized = builder.call(
        "QuantizeLinear",
        input,
        builder.call("CastLike", scales, input)[0],
        builder.call("Cast", zero_points, to=dtype)[0],
        axis=axis,
    )[0]
    if (
        dtype not in _FLOAT8_DTYPES
        and (quant_min, quant_max) != _FULL_QUANTIZATION_RANGE[dtype]
    ):
        # QuantizeLinear saturates to the full range of ``dtype``. PyTorch clamps to the
        # explicit ``quant_min``/``quant_max`` instead, so clamp to match its semantics.
        quantized = _clip(quantized, quant_min, quant_max)
    return quantized


@onnx_impl(
    (
        "quantized_decomposed::dequantize_per_tensor",
        "quantized_decomposed::dequantize_per_tensor.tensor",
        "quantized_decomposed::dequantize_per_tensor.tensor2",
    ),
    opset_introduced=18,
)
def quantized_decomposed_dequantize_per_tensor(
    input: ir.Value,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: int,
    out_dtype: int = -1,
) -> ir.Value:
    builder = current_builder()
    # TODO(justinchuby): Use dtype when we use opset 21
    if isinstance(zero_point, ir.Value):
        zero_point = builder.call("Cast", zero_point, to=dtype)[0]
    else:
        zero_point = common.constant(zero_point, dtype=dtype)
    if isinstance(scale, ir.Value):
        scale = builder.call("Cast", scale, to=ir.DataType.FLOAT)[0]
    dequantized = builder.call("DequantizeLinear", input, scale, zero_point)[0]
    if out_dtype in (-1, None):
        # out_dtype can be None as well
        return dequantized
    if not (out_dtype > 0):
        raise ValueError(f"out_dtype must be -1 or > 0 not {out_dtype}")
    return builder.call("Cast", dequantized, to=out_dtype)[0]


@onnx_impl(
    (
        "quantized_decomposed::dequantize_per_channel",
        "quantized_decomposed::dequantize_per_channel.tensor",
        "quantized_decomposed::dequantize_per_channel.tensor2",
    ),
    opset_introduced=18,
)
def quantized_decomposed_dequantize_per_channel(
    input: ir.Value,
    scales: ir.Value,
    zero_points: ir.Value | None,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: int,
    out_dtype: int = -1,
) -> ir.Value:
    """Affine per channel dequantization for the Tensor using the same quantization
    parameters for each channel/axis to map from quantized values to floating point values.

    Reference:
    https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/fx/_decomposed.py
    ``res = (input - zero_points) * scales`` cast to ``out_dtype`` (float32 by default).
    ``quant_min``/``quant_max``/``dtype`` are metadata only and unused in the computation.
    """
    builder = current_builder()
    # ONNX DequantizeLinear requires a floating point scale and a zero_point that shares
    # the (quantized) input element type. PyTorch passes scales as float64 and zero_points
    # as int64, so cast scales to float32 (the PyTorch default output type) and zero_points
    # to the quantized input type.
    scales = builder.call("Cast", scales, to=ir.DataType.FLOAT)[0]
    if zero_points is not None:
        zero_points = builder.call("Cast", zero_points, to=dtype)[0]
    dequantized = builder.call(
        "DequantizeLinear", input, scales, zero_points, axis=axis
    )[0]
    if out_dtype in (-1, None):
        # PyTorch defaults to float32, which DequantizeLinear already produces.
        return dequantized
    if not (out_dtype > 0):
        raise ValueError(f"out_dtype must be -1 or > 0 not {out_dtype}")
    return builder.call("Cast", dequantized, to=out_dtype)[0]
