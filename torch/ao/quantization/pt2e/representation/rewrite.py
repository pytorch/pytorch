# mypy: allow-untyped-defs
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Tuple

import torch
from torch._higher_order_ops.out_dtype import out_dtype
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.pt2e.export_utils import _WrapperModule
from torch.ao.quantization.pt2e.utils import (
    _get_aten_graph_module_for_pattern,
    _replace_literals_with_existing_placeholders,
    _replace_literals_with_new_placeholders,
    remove_tensor_overload_for_qdq_ops,
)
from torch.fx import GraphModule
from torch.fx.subgraph_rewriter import replace_pattern


__all__ = [
    "reference_representation_rewrite",
]


def _qdq_quantized_linear(
    x_i8,
    x_scale,
    x_zero_point,
    x_quant_min,
    x_quant_max,
    weight_i8,
    weight_scale,
    weight_zero_point,
    weight_quant_min,
    weight_quant_max,
    bias_fp32,
    out_scale,
    out_zero_point,
    out_quant_min,
    out_quant_max,
):
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8
    )
    weight_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        weight_i8,
        weight_scale,
        weight_zero_point,
        weight_quant_min,
        weight_quant_max,
        torch.int8,
    )
    out_fp32 = torch.ops.aten.linear.default(x_fp32, weight_fp32, bias_fp32)
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        out_fp32, out_scale, out_zero_point, out_quant_min, out_quant_max, torch.int8
    )
    return out_i8


def _reference_quantized_linear(
    x_i8,
    x_scale,
    x_zero_point,
    x_quant_min,
    x_quant_max,
    weight_i8,
    weight_scale,
    weight_zero_point,
    weight_quant_min,
    weight_quant_max,
    bias_fp32,
    out_scale,
    out_zero_point,
    out_quant_min,
    out_quant_max,
):
    # without using quant_min/max in clamp, the traced graph will not have quant_mi/max args.
    # This results in failure to match the pattern.
    # Therefore, we call a torch.ops.aten.clamp here
    x_i8 = torch.ops.aten.clamp(x_i8, x_quant_min, x_quant_max)
    weight_i8 = torch.ops.aten.clamp(weight_i8, weight_quant_min, weight_quant_max)

    x_i16 = x_i8.to(torch.int16)
    weight_i16 = weight_i8.to(torch.int16)
    # always set bias to None so that the same representation can work for the case
    # no matter if bias_scale == x_scale * weight_scale or not
    acc_i32 = out_dtype(
        torch.ops.aten.linear.default,
        torch.int32,
        x_i16 - x_zero_point,
        weight_i16 - weight_zero_point,
        None,
    )
    # TODO: change to mul.Scalar
    # Note: we are quantizing bias with these scales without signal from user, but it might be OK
    bias_scale = x_scale * weight_scale
    bias_i32 = out_dtype(torch.ops.aten.div.Tensor, torch.int32, bias_fp32, bias_scale)
    acc_i32 = acc_i32 + bias_i32
    # TODO: change to mul.Scalar when we make x_scale/weight_scale etc. Scalar values
    acc_i32 = (
        out_dtype(
            torch.ops.aten.mul.Tensor,
            torch.int32,
            acc_i32,
            x_scale * weight_scale / out_scale,
        )
        + out_zero_point
    )
    out_i8 = torch.ops.aten.clamp(acc_i32, out_quant_min, out_quant_max).to(torch.int8)
    return out_i8


def _qdq_dynamic_quantized_linear(
    x_fp32,
    x_quant_min,
    x_quant_max,
    x_eps,
    weight_i8,
    weight_scale,
    weight_zero_point,
    weight_quant_min,
    weight_quant_max,
    bias_fp32,
):
    x_scale, x_zero_point = torch.ops.quantized_decomposed.choose_qparams(
        x_fp32, x_quant_min, x_quant_max, x_eps, torch.int8
    )
    x_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        x_fp32, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8
    )
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8
    )
    weight_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        weight_i8,
        weight_scale,
        weight_zero_point,
        weight_quant_min,
        weight_quant_max,
        torch.int8,
    )
    out_fp32 = torch.ops.aten.linear.default(x_fp32, weight_fp32, bias_fp32)
    return out_fp32


def _reference_dynamic_quantized_linear(
    x_fp32,
    x_quant_min,
    x_quant_max,
    x_eps,
    weight_i8,
    weight_scale,
    weight_zero_point,
    weight_quant_min,
    weight_quant_max,
    bias_fp32,
):
    x_scale, x_zero_point = torch.ops.quantized_decomposed.choose_qparams(
        x_fp32, x_quant_min, x_quant_max, x_eps, torch.int8
    )
    # decomposed representation for quantize_per_tensor
    # TODO: use out_dtype(mul, ...) here when the op is ready
    x_fp32 = x_fp32 / x_scale  # fp32
    # round modes might be different here
    # pytorch is rounding to even, which is also common for most of the backends
    x_fp32 = torch.round(x_fp32)  # fp32
    x_i32 = x_fp32.to(dtype=torch.int32)  # int32
    x_i32 = x_i32 + x_zero_point  # int32
    # clamp works for fp32, int32 and int8 dtypes
    x_i32 = torch.clamp(x_i32, x_quant_min, x_quant_max)  # int32
    x_i8 = x_i32.to(dtype=torch.int8)

    weight_i8 = torch.ops.aten.clamp(weight_i8, weight_quant_min, weight_quant_max)

    x_i16 = x_i8.to(torch.int16)
    weight_i16 = weight_i8.to(torch.int16)
    # always set bias to None so that the same representation can work for the case
    # no matter if bias_scale == x_scale * weight_scale or not
    acc_i32 = out_dtype(
        torch.ops.aten.linear.default,
        torch.int32,
        x_i16 - x_zero_point,
        weight_i16 - weight_zero_point,
        None,
    )
    bias_scale = x_scale * weight_scale
    bias_i32 = out_dtype(torch.ops.aten.div.Tensor, torch.int32, bias_fp32, bias_scale)
    acc_i32 = acc_i32 + bias_i32
    out_fp32 = acc_i32 * (x_scale * weight_scale)
    return out_fp32


def _qdq_quantized_conv2d(
    x_i8,
    x_scale,
    x_zero_point,
    x_quant_min,
    x_quant_max,
    weight_i8,
    weight_scale,
    weight_zero_point,
    weight_quant_min,
    weight_quant_max,
    bias_fp32,
    out_scale,
    out_zero_point,
    out_quant_min,
    out_quant_max,
):
    stride = [1, 1]
    padding = [0, 0]
    dilation = [1, 1]
    transposed = False
    output_padding = [0, 0]
    groups = 1
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8
    )
    weight_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        weight_i8,
        weight_scale,
        weight_zero_point,
        weight_quant_min,
        weight_quant_max,
        torch.int8,
    )
    out_fp32 = torch.ops.aten.convolution.default(
        x_fp32,
        weight_fp32,
        bias_fp32,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        out_fp32, out_scale, out_zero_point, out_quant_min, out_quant_max, torch.int8
    )
    return out_i8


def _reference_quantized_conv2d(
    x_i8,
    x_scale,
    x_zero_point,
    x_quant_min,
    x_quant_max,
    weight_i8,
    weight_scale,
    weight_zero_point,
    weight_quant_min,
    weight_quant_max,
    bias_fp32,
    out_scale,
    out_zero_point,
    out_quant_min,
    out_quant_max,
):
    stride = [1, 1]
    padding = [0, 0]
    dilation = [1, 1]
    transposed = False
    output_padding = [0, 0]
    groups = 1
    # without using quant_min/max in clamp, the traced graph will not have quant_mi/max args.
    # This results in failure to match the pattern.
    # Therefore, we call a torch.ops.aten.clamp here
    x_i8 = torch.ops.aten.clamp(x_i8, x_quant_min, x_quant_max)
    weight_i8 = torch.ops.aten.clamp(weight_i8, weight_quant_min, weight_quant_max)

    x_i16 = x_i8.to(torch.int16)
    weight_i16 = weight_i8.to(torch.int16)
    # always set bias to None so that the same representation can work for the case
    # no matter if bias_scale == x_scale * weight_scale or not
    acc_i32 = out_dtype(
        torch.ops.aten.convolution.default,
        torch.int32,
        x_i16 - x_zero_point,
        weight_i16 - weight_zero_point,
        None,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    # Note: we are quantizing bias with these scales without signal from user, but it might be OK
    bias_scale = x_scale * weight_scale
    # bias quantization to int32 uses bias_scale = x_scale * weight_scale due to:
    # Take linear calculation for example
    # Out_(i, j)_fp32 = Sum_(over k)[X_(i, k)_fp32 * W_(i, k)_fp32] + bias_(i)_fp32
    # Represent X, W fp32 as their dequant transforms
    # A_fp32 = (A_q - A_zero_point)/A_scale
    # Out_(i, j)_fp32 = Sum_(over k)[(X_(i, k)_fp32 - X_zp) * X_scale * (W_(i, k)_fp32 - W_zp) * W_scale] + bias_(i)_fp32
    # Factor out X_scale and W_scale
    # Out_(i, j)_fp32 = ((X_scale * W_scale) * Sum_(over k)[(X_(i, k)_fp32 - X_zp) * (W_(i, k)_fp32 - W_zp)]) + bias_(i)_fp32
    # In order to addition of bias_(i)_fp32 inside, we must do
    # Out_(i, j)_fp32 = (X_scale * W_scale) * (Sum_(over k)[(X_(i, k)_fp32 - X_zp) * (W_(i, k)_fp32 - W_zp)] + (1 / (X_scale * W_scale)) * bias_(i)_fp32)W_scale  # noqa: B950
    # Note we had to multiply bias_fp32 qith X_scale * W_scale = bias_scale
    # Thus bias quantization to int32 must be with X_scale * W_scale

    bias_i32 = out_dtype(torch.ops.aten.div.Tensor, torch.int32, bias_fp32, bias_scale)
    # Unsqueeze to match broadcast dims
    # Unfortnuately I cannot do bias_i32.unsqueeze(0) due to literal matching nightmare
    # in graph pattern replacement
    bias_i32 = bias_i32.unsqueeze(-1)
    bias_i32 = bias_i32.unsqueeze(-1)
    acc_i32 = acc_i32 + bias_i32
    # TODO: change to mul.Scalar when we make x_scale/weight_scale etc. Scalar values
    acc_i32 = (
        out_dtype(
            torch.ops.aten.mul.Tensor,
            torch.int32,
            acc_i32,
            x_scale * weight_scale / out_scale,
        )
        + out_zero_point
    )
    out_i8 = torch.ops.aten.clamp(acc_i32, out_quant_min, out_quant_max).to(torch.int8)
    return out_i8


def _qdq_quantized_add_relu(
    x_i8,
    x_scale,
    x_zero_point,
    y_i8,
    y_scale,
    y_zero_point,
    out_scale,
    out_zero_point,
    quant_min,
    quant_max,
):
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x_i8, x_scale, x_zero_point, quant_min, quant_max, torch.int8
    )
    y_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        y_i8, y_scale, y_zero_point, quant_min, quant_max, torch.int8
    )
    out_fp32 = x_fp32 + y_fp32
    out_fp32 = torch.ops.aten.relu(out_fp32)
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        out_fp32, out_scale, out_zero_point, quant_min, quant_max, torch.int8
    )
    return out_i8


def _reference_quantized_add_relu(
    x_i8,
    x_scale,
    x_zero_point,
    y_i8,
    y_scale,
    y_zero_point,
    out_scale,
    out_zero_point,
    quant_min,
    quant_max,
):
    """
    See comments for `_reference_quantized_add` for more information on
    how to derive the formula for out_i8 based on x_i8 and y_i8
    """
    x_i32 = x_i8.to(torch.int32)
    y_i32 = y_i8.to(torch.int32)
    # TODO: change this to mul.Scalar?
    x_i32 = out_dtype(
        torch.ops.aten.mul.Tensor,
        torch.int32,
        (x_i32 - x_zero_point),
        (x_scale / out_scale),
    )
    y_i32 = out_dtype(
        torch.ops.aten.mul.Tensor,
        torch.int32,
        (y_i32 - y_zero_point),
        (y_scale / out_scale),
    )
    out_i32 = x_i32 + y_i32 + out_zero_point
    # out_i32 = torch.ops.aten.clamp(out_i32, out_zero_point)
    out_i8 = torch.ops.aten.clamp(out_i32, out_zero_point, quant_max).to(torch.int8)
    return out_i8


def _qdq_quantized_add(
    x_i8,
    x_scale,
    x_zero_point,
    y_i8,
    y_scale,
    y_zero_point,
    out_scale,
    out_zero_point,
    quant_min,
    quant_max,
):
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x_i8, x_scale, x_zero_point, quant_min, quant_max, torch.int8
    )
    y_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        y_i8, y_scale, y_zero_point, quant_min, quant_max, torch.int8
    )
    out_fp32 = x_fp32 + y_fp32
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        out_fp32, out_scale, out_zero_point, quant_min, quant_max, torch.int8
    )
    return out_i8


def _reference_quantized_add(
    x_i8,
    x_scale,
    x_zero_point,
    y_i8,
    y_scale,
    y_zero_point,
    out_scale,
    out_zero_point,
    quant_min,
    quant_max,
):
    """
        # How to Derive the formula for out_i8 based on x_i8 and y_i8
        # (since quantized add takes x_i8, y_i8 and their quantization parameters, and produce an out_i8)

        # out_i8 is quantized output, we can write down the formula for it first:
    out_i8 = out_f32 / out_scale + out_zero_point           (1)

        # then out_fp32 is computed from x_f32 + y_f32, and the x_fp32 and y_fp32 are the dequantized x_i8 and y_i8
        out_f32 = x_f32 + y_f32           (2)
        x_fp32 = (x_i8 - x_zero_point) * x_scale         (3)
        y_fp32 = (y_i8 - y_zero_point) * y_scale         (4)

        # applying the above fomula to the out_i8 equation we can get the following:
        out_i8 = out_fp32 / out_scale + out_zero_point             # (1)
           = (x_f32 + y_f32) / out_scale + out_zero_point      # applying (2) to substitute out_fp32 with x_fp32 + y_fp32
           = ((x_i8 - x_zero_point) * x_scale + (y_i8 - y_zero_point) * y_scale) / out_scale + out_zero_point  # apply (3) and (4)
    """
    x_i32 = x_i8.to(torch.int32)
    y_i32 = y_i8.to(torch.int32)
    # TODO: use out_dtype op
    x_i32 = torch.round((x_scale / out_scale) * (x_i32 - x_zero_point)).to(torch.int32)
    y_i32 = torch.round((y_scale / out_scale) * (y_i32 - y_zero_point)).to(torch.int32)
    out_i32 = x_i32 + y_i32 + out_zero_point
    quant_min = -128
    quant_max = 127
    out_i8 = torch.ops.aten.clamp(out_i32, quant_min, quant_max).to(torch.int8)
    return out_i8


def _qdq_quantized_max_pool2d(
    x_i8,
    x_scale,
    x_zero_point,
    x_quant_min,
    x_quant_max,
    out_scale,
    out_zero_point,
    out_quant_min,
    out_quant_max,
):
    kernel_size = 1
    stride = 1
    padding = 0
    dilation = 1
    ceil_mode = False
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8
    )
    out_fp32, _ = torch.ops.aten.max_pool2d_with_indices.default(
        x_fp32, kernel_size, stride, padding, dilation, ceil_mode
    )
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        out_fp32, out_scale, out_zero_point, out_quant_min, out_quant_max, torch.int8
    )
    return out_i8


def _reference_quantized_max_pool2d(
    x_i8,
    x_scale,
    x_zero_point,
    x_quant_min,
    x_quant_max,
    out_scale,
    out_zero_point,
    out_quant_min,
    out_quant_max,
):
    kernel_size = 1
    stride = 1
    padding = 0
    dilation = 1
    ceil_mode = False
    # to preserve x_quant_min, x_quant_max in the graph for pattern matching
    x_i8 = torch.clamp(x_i8, x_quant_min, x_quant_max)
    x_i32 = x_i8.to(torch.int32)
    out_i32, _ = torch.ops.aten.max_pool2d_with_indices.default(
        x_i32 - x_zero_point, kernel_size, stride, padding, dilation, ceil_mode
    )
    out_fp32 = out_i32 * (x_scale / out_scale) + out_zero_point
    out_fp32 = torch.clamp(out_fp32, out_quant_min, out_quant_max)
    out_i8 = out_fp32.to(torch.int8)
    return out_i8


def _quantize_per_tensor_int8(x_fp32, scale, zero_point, quant_min, quant_max):
    x = torch.ops.quantized_decomposed.quantize_per_tensor(
        x_fp32, scale, zero_point, quant_min, quant_max, torch.int8
    )
    return x


def _reference_quantize_per_tensor_int8(
    x_fp32, scale, zero_point, quant_min, quant_max
):
    # TODO: use out_dtype(mul, ...) here when the op is ready
    x = x_fp32 / scale  # fp32
    # round modes might be different here
    # pytorch is rounding to even, which is also common for most of the backends
    x = torch.round(x)  # fp32
    x = x.to(dtype=torch.int32)  # int32
    x = x + zero_point  # int32
    # clamp works for fp32, int32 and int8 dtypes
    x = torch.clamp(x, quant_min, quant_max)  # int32
    x = x.to(dtype=torch.int8)
    return x


def _dequantize_per_tensor_int8(x_i8, scale, zero_point, quant_min, quant_max):
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(
        x_i8, scale, zero_point, quant_min, quant_max, torch.int8
    )
    return x_fp32


def _reference_dequantize_per_tensor_int8(
    x_i8, scale, zero_point, quant_min, quant_max
):
    # without using quant_min/max in clamp, the traced graph will not have quant_mi/max args.
    # This results in failure to match the pattern.
    # Therefore, we call a torch.ops.aten.clamp here
    x_i8 = torch.ops.aten.clamp(x_i8, quant_min, quant_max)
    # TODO: use out_dtype op
    # note: x_i8.to(torch.int32) does not work here
    # TODO: debug the implementation later when torchdynamo time out issue is resolved
    return ((x_i8.to(torch.float32) - zero_point) * scale).to(dtype=torch.float32)


def _quantize_per_channel_int8(
    x_fp32, scales, zero_points, ch_axis, quant_min, quant_max
):
    out_i8 = torch.ops.quantized_decomposed.quantize_per_channel(
        x_fp32, scales, zero_points, ch_axis, quant_min, quant_max, torch.int8
    )
    return out_i8


def _reference_quantize_per_channel_int8(
    x_fp32, scales, zero_points, ch_axis, quant_min, quant_max
):
    x_fp32 = torch.transpose(x_fp32, ch_axis, -1)
    out_i32 = torch.ops.aten.clamp(
        torch.round(x_fp32 / scales).to(torch.int32) + zero_points, quant_min, quant_max
    )
    out_i32 = torch.transpose(out_i32, ch_axis, -1)
    return out_i32.to(torch.int8)


def _dequantize_per_channel_int8(
    x_i8, scales, zero_points, ch_axis, quant_min, quant_max
):
    # the following will be replaced as placeholders
    out_fp32 = torch.ops.quantized_decomposed.dequantize_per_channel(
        x_i8, scales, zero_points, ch_axis, quant_min, quant_max, torch.int8
    )
    return out_fp32


def _reference_dequantize_per_channel_int8(
    x_i8, scales, zero_points, ch_axis, quant_min, quant_max
):
    # the following will be replaced as placeholders
    # in order to preserve the quant_min/quant_max args for pattern matching (e.g. matching for int4 quantized ops)
    # we call a torch.ops.aten.clamp here
    x_i8 = torch.ops.aten.clamp(x_i8, quant_min, quant_max)
    x_i8 = torch.transpose(x_i8, ch_axis, -1)
    x_i32 = x_i8.to(torch.int32)
    out_fp32 = (x_i32 - zero_points).to(torch.float) * scales
    out_fp32 = torch.transpose(out_fp32, ch_axis, -1)
    return out_fp32


def _replace_ph_qdq_per_channel_replacement(gm: torch.fx.GraphModule):
    return _replace_literals_with_existing_placeholders(
        gm, exclude_literals=[-1], literal_to_ph_idx={1: 3, -128: 4, 127: 5}
    )


@dataclass
class _RewriteInfo:
    """Data needed for rewrite, this includes example inputs, pattern and replacement functions
    and post transformation functions for the exported pattern and replacement GraphModule
    """

    # example inputs used for exporting the pattern into GraphModule
    example_inputs: Tuple[Any, ...]
    pattern: Callable
    replacement: Callable
    # post transformation on the exported pattern and replacement GraphModule
    pattern_post_trans: Optional[Callable[[GraphModule], GraphModule]] = None
    replacement_post_trans: Optional[Callable[[GraphModule], GraphModule]] = None


def reference_representation_rewrite(model: GraphModule) -> GraphModule:
    _QUANTIZED_LINEAR_EXAMPLE_INPUTS = (
        torch.randint(-128, 127, (2, 5), dtype=torch.int8),
        torch.randn(1, dtype=torch.float),
        torch.zeros(1, dtype=torch.int),
        torch.tensor([-128], dtype=torch.int),
        torch.tensor([127], dtype=torch.int),
        torch.randint(-128, 127, (5, 5), dtype=torch.int8),
        torch.randn(1, dtype=torch.float),
        torch.zeros(1, dtype=torch.int),
        torch.tensor([-127], dtype=torch.int),
        torch.tensor([127], dtype=torch.int),
        torch.randn(1, dtype=torch.float),
        torch.randn(1, dtype=torch.float),
        torch.zeros(1, dtype=torch.int),
        torch.tensor([-128], dtype=torch.int),
        torch.tensor([127], dtype=torch.int),
    )

    _DYNAMIC_QUANTIZED_LINEAR_EXAMPLE_INPUTS = (
        torch.randn((2, 5), dtype=torch.float),
        -128,
        127,
        torch.finfo(torch.float32).eps,
        torch.randint(-128, 127, (5, 5), dtype=torch.int8),
        torch.randn(1, dtype=torch.float),
        torch.zeros(1, dtype=torch.int),
        torch.tensor([-127], dtype=torch.int),
        torch.tensor([127], dtype=torch.int),
        torch.randn(1, dtype=torch.float),
    )

    _QUANTIZED_CONV2d_EXAMPLE_INPUTS = (
        torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),
        torch.randn(1, dtype=torch.float),
        torch.zeros(1, dtype=torch.int),
        torch.tensor([-128], dtype=torch.int),
        torch.tensor([127], dtype=torch.int),
        torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),
        torch.randn(1, dtype=torch.float),
        torch.zeros(1, dtype=torch.int),
        torch.tensor([-127], dtype=torch.int),
        torch.tensor([127], dtype=torch.int),
        torch.randn(1, dtype=torch.float),
        torch.randn(1, dtype=torch.float),
        torch.zeros(1, dtype=torch.int),
        torch.tensor([-128], dtype=torch.int),
        torch.tensor([127], dtype=torch.int),
    )

    _QUANTIZED_ADD_OR_ADD_RELU_EXAMPLE_INPUTS = (
        torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),
        torch.randn(1, dtype=torch.float),
        torch.zeros(1, dtype=torch.int),
        torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),
        torch.randn(1, dtype=torch.float),
        torch.zeros(1, dtype=torch.int),
        torch.randn(1, dtype=torch.float),
        torch.zeros(1, dtype=torch.int),
        torch.tensor([-128], dtype=torch.int),
        torch.tensor([127], dtype=torch.int),
    )

    _QUANTIZED_MAX_POOL2D_EXAMPLE_INPUTS = (
        torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),
        torch.randn(1, dtype=torch.float),
        torch.zeros(1, dtype=torch.int),
        torch.tensor([-128], dtype=torch.int),
        torch.tensor([127], dtype=torch.int),
        torch.randn(1, dtype=torch.float),
        torch.zeros(1, dtype=torch.int),
        torch.tensor([-128], dtype=torch.int),
        torch.tensor([127], dtype=torch.int),
    )

    _QUANTIZE_PER_TENSOR_INT8_EXAMPLE_INPUTS = (
        torch.randn(1, 3, 3, 3, dtype=torch.float),
        torch.randn(1, dtype=torch.float),
        torch.zeros(1, dtype=torch.int),
        torch.tensor([-128], dtype=torch.int),
        torch.tensor([127], dtype=torch.int),
    )

    _DEQUANTIZE_PER_TENSOR_INT8_EXAMPLE_INPUTS = (
        torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),
        torch.randn(1, dtype=torch.float),
        torch.zeros(1, dtype=torch.int),
        torch.tensor([-128], dtype=torch.int),
        torch.tensor([127], dtype=torch.int),
    )

    _QUANTIZE_PER_CHANNEL_INT8_EXAMPLE_INPUTS = (
        torch.randn(1, 3, 3, 3, dtype=torch.float),
        torch.randn(3, dtype=torch.float),
        torch.zeros(3, dtype=torch.int),
        1,
        -128,
        127,
    )

    _DEQUANTIZE_PER_CHANNEL_INT8_EXAMPLE_INPUTS = (
        torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),
        torch.randn(3, dtype=torch.float),
        torch.zeros(3, dtype=torch.int),
        1,
        -128,
        127,
    )

    _REWRITE_INFO_LIST = [
        _RewriteInfo(
            _DYNAMIC_QUANTIZED_LINEAR_EXAMPLE_INPUTS,
            _WrapperModule(_qdq_dynamic_quantized_linear),
            _WrapperModule(_reference_dynamic_quantized_linear),
            partial(
                _replace_literals_with_existing_placeholders,
                literal_to_ph_idx={-128: 1, 127: 2, torch.finfo(torch.float32).eps: 3},
            ),
            partial(
                _replace_literals_with_existing_placeholders,
                literal_to_ph_idx={-128: 1, 127: 2, torch.finfo(torch.float32).eps: 3},
            ),
        ),
        _RewriteInfo(
            _QUANTIZED_LINEAR_EXAMPLE_INPUTS,
            _WrapperModule(_qdq_quantized_linear),
            _WrapperModule(_reference_quantized_linear),
            _replace_literals_with_new_placeholders,
            _replace_literals_with_new_placeholders,
        ),
        _RewriteInfo(
            _QUANTIZED_CONV2d_EXAMPLE_INPUTS,
            _WrapperModule(_qdq_quantized_conv2d),
            _WrapperModule(_reference_quantized_conv2d),
            partial(_replace_literals_with_new_placeholders, exclude_literals=[-1]),
            partial(_replace_literals_with_new_placeholders, exclude_literals=[-1]),
        ),
        _RewriteInfo(
            _QUANTIZED_ADD_OR_ADD_RELU_EXAMPLE_INPUTS,
            _WrapperModule(_qdq_quantized_add_relu),
            _WrapperModule(_reference_quantized_add_relu),
        ),
        _RewriteInfo(
            _QUANTIZED_ADD_OR_ADD_RELU_EXAMPLE_INPUTS,
            _WrapperModule(_qdq_quantized_add),
            _WrapperModule(_reference_quantized_add),
        ),
        _RewriteInfo(
            _QUANTIZED_MAX_POOL2D_EXAMPLE_INPUTS,
            _WrapperModule(_qdq_quantized_max_pool2d),
            _WrapperModule(_reference_quantized_max_pool2d),
            _replace_literals_with_new_placeholders,
            _replace_literals_with_new_placeholders,
        ),
        _RewriteInfo(
            _QUANTIZE_PER_TENSOR_INT8_EXAMPLE_INPUTS,
            _WrapperModule(_quantize_per_tensor_int8),
            _WrapperModule(_reference_quantize_per_tensor_int8),
        ),
        _RewriteInfo(
            _DEQUANTIZE_PER_TENSOR_INT8_EXAMPLE_INPUTS,
            _WrapperModule(_dequantize_per_tensor_int8),
            _WrapperModule(_reference_dequantize_per_tensor_int8),
        ),
        _RewriteInfo(
            _QUANTIZE_PER_CHANNEL_INT8_EXAMPLE_INPUTS,
            _WrapperModule(_quantize_per_channel_int8),
            _WrapperModule(_reference_quantize_per_channel_int8),
            _replace_ph_qdq_per_channel_replacement,
            _replace_ph_qdq_per_channel_replacement,
        ),
        _RewriteInfo(
            _DEQUANTIZE_PER_CHANNEL_INT8_EXAMPLE_INPUTS,
            _WrapperModule(_dequantize_per_channel_int8),
            _WrapperModule(_reference_dequantize_per_channel_int8),
            _replace_ph_qdq_per_channel_replacement,
            _replace_ph_qdq_per_channel_replacement,
        ),
    ]

    remove_tensor_overload_for_qdq_ops(model)

    for rewrite_info in _REWRITE_INFO_LIST:
        example_inputs = rewrite_info.example_inputs
        pattern = rewrite_info.pattern
        replacement = rewrite_info.replacement
        pattern_post_trans = rewrite_info.pattern_post_trans
        replacement_post_trans = rewrite_info.replacement_post_trans
        pattern = _get_aten_graph_module_for_pattern(pattern, example_inputs)  # type: ignore[arg-type, assignment]
        remove_tensor_overload_for_qdq_ops(pattern)  # type: ignore[arg-type]
        replacement = _get_aten_graph_module_for_pattern(replacement, example_inputs)  # type: ignore[arg-type, assignment]
        remove_tensor_overload_for_qdq_ops(replacement)  # type: ignore[arg-type]
        if pattern_post_trans:
            pattern = pattern_post_trans(pattern)
        if replacement_post_trans:
            replacement = replacement_post_trans(replacement)
        pattern.recompile()  # type: ignore[attr-defined]
        replacement.recompile()  # type: ignore[attr-defined]
        replace_pattern(model, pattern, replacement)

    return model
