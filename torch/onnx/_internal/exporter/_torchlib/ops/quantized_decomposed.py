"""torch.ops.quantized_decomposed operators."""

# mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value"
# pyrefly: ignore-errors
# ruff: noqa: TC001

from __future__ import annotations

from onnxscript.onnx_opset import opset18 as op

import torch
import torch.ao.quantization.fx._decomposed  # registers the quantized_decomposed ops
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.onnx._internal.exporter._torchlib._tensor_typing import TensorType
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


quantized_decomposed = torch.ops.quantized_decomposed

# Full quantization range of every dtype the quantized_decomposed ops accept
# (torch/ao/quantization/fx/_decomposed.py). int32 is the only one ONNX
# QuantizeLinear cannot emit (e.g. bias quantization) and is lowered to
# primitive ops instead.
_FULL_QUANTIZATION_RANGE = {
    ir.DataType.INT8: (-128, 127),
    ir.DataType.UINT8: (0, 255),
    ir.DataType.INT16: (-(2**15), 2**15 - 1),
    ir.DataType.UINT16: (0, 2**16 - 1),
    ir.DataType.INT32: (-(2**31), 2**31 - 1),
    ir.DataType.FLOAT8E4M3FN: (-448, 448),
    ir.DataType.FLOAT8E5M2: (-57344, 57344),
}

# Clip accepts every numeric tensor type except float8, so a narrower than
# dtype float8 range keeps QuantizeLinear's saturation instead of clamping.
_FLOAT8_DTYPES = frozenset(
    (
        ir.DataType.FLOAT8E4M3FN,
        ir.DataType.FLOAT8E4M3FNUZ,
        ir.DataType.FLOAT8E5M2,
        ir.DataType.FLOAT8E5M2FNUZ,
    )
)


@onnx_impl(
    (
        quantized_decomposed.quantize_per_tensor.default,
        quantized_decomposed.quantize_per_tensor.tensor,
        quantized_decomposed.quantize_per_tensor.tensor2,
    ),
    trace_only=True,
)
def quantized_decomposed_quantize_per_tensor(
    input: TensorType,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: int,
) -> TensorType:
    """quantize_per_tensor(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype) -> Tensor"""

    if dtype == ir.DataType.INT32:
        # PyTorch reference: clamp(round(input * (1 / scale)) + zero_point, quant_min, quant_max).to(int32).
        # Reciprocal+Mul, not Div: they round differently in float32.
        if input.dtype in (ir.DataType.FLOAT16, ir.DataType.BFLOAT16):
            input = op.Cast(input, to=ir.DataType.FLOAT)
        scaled = op.Mul(input, op.Reciprocal(op.CastLike(scale, input)))
        quantized = op.Add(op.Round(scaled), op.CastLike(zero_point, input))
    else:
        quantized = op.QuantizeLinear(
            input, op.CastLike(scale, input), op.Cast(zero_point, to=dtype)
        )
    if (
        dtype not in _FLOAT8_DTYPES
        and (quant_min, quant_max) != _FULL_QUANTIZATION_RANGE[dtype]
    ):
        # QuantizeLinear saturates to the full dtype range; PyTorch clamps to
        # quant_min/quant_max, which may be narrower.
        quantized = op.Clip(
            quantized,
            op.CastLike(quant_min, quantized),
            op.CastLike(quant_max, quantized),
        )
    return quantized if dtype != ir.DataType.INT32 else op.Cast(quantized, to=dtype)


@onnx_impl(quantized_decomposed.quantize_per_channel.default, trace_only=True)
def quantized_decomposed_quantize_per_channel(
    input: TensorType,
    scales: TensorType,
    zero_points: TensorType,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: int,
) -> TensorType:
    """quantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, int quant_min, int quant_max, ScalarType dtype) -> Tensor"""

    if dtype == ir.DataType.INT32:
        shape = input.shape
        if shape is None:
            raise NotImplementedError(
                "Exporting int32 quantize_per_channel requires a static input "
                "rank to broadcast scales and zero_points along 'axis'"
            )
        # The PyTorch reference computes round(input * (1 / scales)) in the type
        # promoted from input and scales: float64 for the standard float64
        # scales, float32 otherwise.
        compute_dtype = (
            ir.DataType.DOUBLE
            if scales.dtype == ir.DataType.DOUBLE
            else ir.DataType.FLOAT
        )
        if input.dtype != compute_dtype:
            input = op.Cast(input, to=compute_dtype)
        # Reshape the 1-D scales/zero_points to broadcast along 'axis' and apply
        # clamp(round(input * (1 / scales)) + zero_points, quant_min, quant_max).to(int32)
        axes = [i for i in range(len(shape)) if i != axis]
        scales = op.Unsqueeze(op.Reciprocal(op.CastLike(scales, input)), axes=axes)
        zero_points = op.Unsqueeze(op.CastLike(zero_points, input), axes=axes)
        quantized = op.Add(op.Round(op.Mul(input, scales)), zero_points)
    else:
        quantized = op.QuantizeLinear(
            input,
            op.CastLike(scales, input),
            op.Cast(zero_points, to=dtype),
            axis=axis,
        )
    if (
        dtype not in _FLOAT8_DTYPES
        and (quant_min, quant_max) != _FULL_QUANTIZATION_RANGE[dtype]
    ):
        # QuantizeLinear saturates to the full dtype range; PyTorch clamps to
        # quant_min/quant_max, which may be narrower.
        quantized = op.Clip(
            quantized,
            op.CastLike(quant_min, quantized),
            op.CastLike(quant_max, quantized),
        )
    return quantized if dtype != ir.DataType.INT32 else op.Cast(quantized, to=dtype)
