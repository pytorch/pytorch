"""torch.ops.quantized_decomposed operators."""

# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value"
# pyrefly: ignore-errors

from __future__ import annotations

import torch
import torch.ao.quantization.fx._decomposed  # registers the quantized_decomposed ops
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.onnx._internal.exporter._ir_builder import current_builder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


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


@onnx_impl(quantized_decomposed.quantize_per_channel.default)
def quantized_decomposed_quantize_per_channel(
    input: ir.Value,
    scales: ir.Value,
    zero_points: ir.Value,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: int,
) -> ir.Value:
    """quantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, int quant_min, int quant_max, ScalarType dtype) -> Tensor"""
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
        # QuantizeLinear saturates to the full dtype range; narrower ranges need a Clip.
        quantized = _clip(quantized, quant_min, quant_max)
    return quantized
