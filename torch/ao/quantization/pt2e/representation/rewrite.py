import torch
from torch.fx import GraphModule
from ..utils import (
    get_aten_graph_module,
    remove_tensor_overload_for_qdq_ops,
    replace_literals_with_new_placeholders,
    replace_literals_with_existing_placeholders,
)
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.fx.subgraph_rewriter import replace_pattern
from torch._higher_order_ops.out_dtype import out_dtype

__all__ = [
    "reference_representation_rewrite",
]

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

def _qdq_quantized_conv2d(x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, bias_fp32, out_scale, out_zero_point, out_quant_min, out_quant_max):
    stride = [1, 1]
    padding = [0, 0]
    dilation = [1, 1]
    transposed = False
    output_padding = [0, 0]
    groups = 1
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, torch.int8)
    weight_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, torch.int8)
    out_fp32 = torch.ops.aten.convolution.default(x_fp32, weight_fp32, bias_fp32, stride, padding, dilation, transposed, output_padding, groups)
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(out_fp32, out_scale, out_zero_point, out_quant_min, out_quant_max, torch.int8)
    return out_i8

def _reference_quantized_conv2d(x_i8, x_scale, x_zero_point, x_quant_min, x_quant_max, weight_i8, weight_scale, weight_zero_point, weight_quant_min, weight_quant_max, bias_fp32, out_scale, out_zero_point, out_quant_min, out_quant_max):
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
    # always set bias to None so that the same representation can work for the case no matter if bias_scale == x_scale * weight_scale or not
    acc_i32 = out_dtype(torch.ops.aten.convolution.default, torch.int32, x_i16 - x_zero_point, weight_i16 - weight_zero_point, None, stride, padding, dilation, transposed, output_padding, groups)
    # TODO: change to mul.Scalar when we make x_scale/weight_scale etc. Scalar values
    acc_i32 = out_dtype(torch.ops.aten.mul.Tensor, torch.int32, acc_i32, x_scale * weight_scale / out_scale)
    # TODO: change to mul.Scalar
    # Note: we are quantizing bias with these scales without signal from user, but it might be OK
    bias_scale = x_scale * weight_scale
    bias_i32 = out_dtype(torch.ops.aten.mul.Tensor, torch.int32, bias_fp32, bias_scale / out_scale)
    out_i8 = torch.ops.aten.clamp(acc_i32, out_quant_min, out_quant_max).to(torch.int8)
    return out_i8


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

def _qdq_quantized_add_relu(
    x_i8, x_scale, x_zero_point, y_i8, y_scale, y_zero_point,
    out_scale, out_zero_point, quant_min, quant_max
):
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(x_i8, x_scale, x_zero_point, quant_min, quant_max, torch.int8)
    y_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(y_i8, y_scale, y_zero_point, quant_min, quant_max, torch.int8)
    out_fp32 = x_fp32 + y_fp32
    out_fp32 = torch.ops.aten.relu(out_fp32)
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        out_fp32, out_scale, out_zero_point, quant_min, quant_max, torch.int8
    )
    return out_i8

def _reference_quantized_add_relu(
    x_i8, x_scale, x_zero_point, y_i8, y_scale, y_zero_point,
    out_scale, out_zero_point, quant_min, quant_max
):
    """
    See comments for `_reference_quantized_add` for more information on
    how to derive the formula for out_i8 based on x_i8 and y_i8
    """
    x_i32 = x_i8.to(torch.int32)
    y_i32 = y_i8.to(torch.int32)
    # TODO: change this to mul.Scalar?
    x_i32 = out_dtype(torch.ops.aten.mul.Tensor, torch.int32, (x_i32 - x_zero_point), (x_scale / out_scale))
    y_i32 = out_dtype(torch.ops.aten.mul.Tensor, torch.int32, (y_i32 - y_zero_point), (y_scale / out_scale))
    out_i32 = x_i32 + y_i32 + out_zero_point
    # out_i32 = torch.ops.aten.clamp(out_i32, out_zero_point)
    out_i8 = torch.ops.aten.clamp(out_i32, out_zero_point, quant_max).to(torch.int8)
    return out_i8

def _qdq_quantized_add(x_i8, x_scale, x_zero_point, y_i8, y_scale, y_zero_point, out_scale, out_zero_point, quant_min, quant_max):
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(x_i8, x_scale, x_zero_point, quant_min, quant_max, torch.int8)
    y_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(y_i8, y_scale, y_zero_point, quant_min, quant_max, torch.int8)
    out_fp32 = x_fp32 + y_fp32
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        out_fp32, out_scale, out_zero_point, quant_min, quant_max, torch.int8
    )
    return out_i8

def _reference_quantized_add(
    x_i8, x_scale, x_zero_point, y_i8, y_scale, y_zero_point,
    out_scale, out_zero_point, quant_min, quant_max
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

_QUANTIZED_MAX_POOL2D_EXAMPLE_INPUTS = (
    torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),
    torch.randn(1, dtype=torch.float),
    torch.zeros(1, dtype=torch.int),
    torch.randn(1, dtype=torch.float),
    torch.zeros(1, dtype=torch.int),
    torch.tensor([-128], dtype=torch.int),
    torch.tensor([127], dtype=torch.int),
)

def _qdq_quantized_max_pool2d(x_i8, x_scale, x_zero_point, out_scale, out_zero_point, quant_min, quant_max):
    kernel_size = 1
    stride = 1
    padding = 0
    dilation = 1
    ceil_mode = False
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(x_i8, x_scale, x_zero_point, quant_min, quant_max, torch.int8)
    out_fp32, _ = torch.ops.aten.max_pool2d_with_indices.default(x_fp32, kernel_size, stride, padding, dilation, ceil_mode)
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(out_fp32, out_scale, out_zero_point, quant_min, quant_max, torch.int8)
    return out_i8

def _reference_quantized_max_pool2d(x_i8, x_scale, x_zero_point, out_scale, out_zero_point, quant_min, quant_max):
    kernel_size = 1
    stride = 1
    padding = 0
    dilation = 1
    ceil_mode = False
    acc_i8, _ = torch.ops.aten.max_pool2d_with_indices.default(x_i8, kernel_size, stride, padding, dilation, ceil_mode)
    acc_i32 = acc_i8.to(torch.int32)
    # TODO: use mul.Scalar, need to change how we handle literal args
    output_i32 = out_dtype(torch.ops.aten.mul.Tensor, torch.int32, acc_i32, (x_scale / out_scale))
    output_i8 = output_i32 - (x_zero_point * x_scale / out_scale + out_zero_point)
    output_i8 = output_i8.to(torch.int8)
    return output_i8

_QUANTIZED_ADAPTIVE_AVG_POOL2D_EXAMPLE_INPUTS = (
    torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),
    torch.randn(1, dtype=torch.float),
    torch.zeros(1, dtype=torch.int),
    torch.randn(1, dtype=torch.float),
    torch.zeros(1, dtype=torch.int),
    torch.tensor([-128], dtype=torch.int),
    torch.tensor([127], dtype=torch.int),
)

def _qdq_quantized_adaptive_avg_pool2d(x_i8, x_scale, x_zero_point, out_scale, out_zero_point, quant_min, quant_max):
    output_size = (1, 1)
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(x_i8, x_scale, x_zero_point, quant_min, quant_max, torch.int8)
    out_fp32 = torch.ops.aten.adaptive_avg_pool2d(x_fp32, output_size)
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(out_fp32, out_scale, out_zero_point, quant_min, quant_max, torch.int8)
    return out_i8

def _reference_quantized_adaptive_avg_pool2d(x_i8, x_scale, x_zero_point, out_scale, out_zero_point, quant_min, quant_max):
    output_size = (1, 1)
    x_i32 = x_i8.to(torch.int32)
    # TODO: use mul.Scalar, need to change how literal args are handled
    x_i32 = out_dtype(torch.ops.aten.mul.Tensor, torch.int32, x_i32, (x_scale / out_scale))
    acc_i32 = torch.ops.aten._adaptive_avg_pool2d(x_i32, output_size)
    acc_i32 = acc_i32 - x_zero_point * x_scale / out_scale + out_zero_point # int32 constant
    out_i8 = torch.ops.aten.clamp(acc_i32, quant_min, quant_max).to(torch.int8)
    return out_i8

_QUANTIZE_PER_TENSOR_INT8_EXAMPLE_INPUTS = (
    torch.randn(1, 3, 3, 3, dtype=torch.float),
    torch.randn(1, dtype=torch.float),
    torch.zeros(1, dtype=torch.int),
    torch.tensor([-128], dtype=torch.int),
    torch.tensor([127], dtype=torch.int),
)

def _quantize_per_tensor_int8(x_fp32, scale, zero_point, quant_min, quant_max):
    x = torch.ops.quantized_decomposed.quantize_per_tensor(x_fp32, scale, zero_point, quant_min, quant_max, torch.int8)
    return x

def _reference_quantize_per_tensor_int8(x_fp32, scale, zero_point, quant_min, quant_max):
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

_DEQUANTIZE_PER_TENSOR_INT8_EXAMPLE_INPUTS = (
    torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),
    torch.randn(1, dtype=torch.float),
    torch.zeros(1, dtype=torch.int),
    torch.tensor([-128], dtype=torch.int),
    torch.tensor([127], dtype=torch.int),
)

def _dequantize_per_tensor_int8(x_i8, scale, zero_point, quant_min, quant_max):
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(x_i8, scale, zero_point, quant_min, quant_max, torch.int8)
    return x_fp32

def _reference_dequantize_per_tensor_int8(x_i8, scale, zero_point, quant_min, quant_max):
    # without using quant_min/max in clamp, the traced graph will not have quant_mi/max args.
    # This results in failure to match the pattern.
    # Therefore, we call a torch.ops.aten.clamp here
    x_i8 = torch.ops.aten.clamp(x_i8, quant_min, quant_max)
    # TODO: use out_dtype op
    # note: x_i8.to(torch.int32) does not work here
    # TODO: debug the implementation later when torchdynamo time out issue is resolved
    return ((x_i8.to(torch.float32) - zero_point) * scale).to(dtype=torch.float32)

_QUANTIZE_PER_CHANNEL_INT8_EXAMPLE_INPUTS = (
    torch.randn(1, 3, 3, 3, dtype=torch.float),
    torch.randn(3, dtype=torch.float),
    torch.zeros(3, dtype=torch.int),
    1,
    -128,
    127,
)

def _quantize_per_channel_int8(x_fp32, scales, zero_points, ch_axis, quant_min, quant_max):
    out_i8 = torch.ops.quantized_decomposed.quantize_per_channel(x_fp32, scales, zero_points, ch_axis, quant_min, quant_max, torch.int8)
    return out_i8

def _reference_quantize_per_channel_int8(x_fp32, scales, zero_points, ch_axis, quant_min, quant_max):
    x_fp32 = torch.transpose(x_fp32, ch_axis, -1)
    out_i32 = torch.ops.aten.clamp(torch.round(x_fp32 / scales).to(torch.int32) + zero_points, quant_min, quant_max)
    out_i32 = torch.transpose(out_i32, ch_axis, -1)
    return out_i32.to(torch.int8)

_DEQUANTIZE_PER_CHANNEL_INT8_EXAMPLE_INPUTS = (
    torch.randint(-128, 127, (1, 3, 3, 3), dtype=torch.int8),
    torch.randn(3, dtype=torch.float),
    torch.zeros(3, dtype=torch.int),
    1,
    -128,
    127,
)

def _dequantize_per_channel_int8(x_i8, scales, zero_points, ch_axis, quant_min, quant_max):
    # the following will be replaced as placeholders
    out_fp32 = torch.ops.quantized_decomposed.dequantize_per_channel(x_i8, scales, zero_points, ch_axis, quant_min, quant_max, torch.int8)
    return out_fp32

def _reference_dequantize_per_channel_int8(x_i8, scales, zero_points, ch_axis, quant_min, quant_max):
    # the following will be replaced as placeholders
    # in order to preserve the quant_min/quant_max args for pattern matching (e.g. matching for int4 quantized ops)
    # we call a torch.ops.aten.clamp here
    x_i8 = torch.ops.aten.clamp(x_i8, quant_min, quant_max)
    x_i8 = torch.transpose(x_i8, ch_axis, -1)
    x_i32 = x_i8.to(torch.int32)
    out_fp32 = (x_i32 - zero_points) * scales
    out_fp32 = torch.transpose(out_fp32, ch_axis, -1)
    return out_fp32

def _replace_ph_qdq_per_channel_replacement(gm: torch.fx.GraphModule):
    return replace_literals_with_existing_placeholders(
        gm,
        exclude_literals=[-1],
        literal_to_ph_idx={1: 3, -128: 4, 127:5}
    )

# (example inputs, pattern, replacement, post_transformation_pattern, post_transformation_replacement)
_EXAMPLE_INPUTS_PATTERN_AND_REPLACEMENTS = [
<<<<<<< HEAD
    (_QUANTIZED_ADD_OR_ADD_RELU_EXAMPLE_INPUTS, _qdq_quantized_add_relu, _reference_quantized_add_relu, None, None),
    (_QUANTIZED_ADD_OR_ADD_RELU_EXAMPLE_INPUTS, _qdq_quantized_add, _reference_quantized_add, None, None),
=======
    (_QUANTIZED_CONV2d_EXAMPLE_INPUTS, _qdq_quantized_conv2d, _reference_quantized_conv2d, replace_literals_with_new_placeholders, replace_literals_with_new_placeholders),
    (_QUANTIZED_ADD_EXAMPLE_INPUTS, _qdq_quantized_add_relu, _reference_quantized_add_relu, None, None),
    (_QUANTIZED_ADD_EXAMPLE_INPUTS, _qdq_quantized_add, _reference_quantized_add, None, None),
>>>>>>> bdad4feef10 ([quant][pt2e] Add reference representation for quantized conv2d)
    (_QUANTIZED_MAX_POOL2D_EXAMPLE_INPUTS, _qdq_quantized_max_pool2d, _reference_quantized_max_pool2d, replace_literals_with_new_placeholders, replace_literals_with_new_placeholders),
    (_QUANTIZED_ADAPTIVE_AVG_POOL2D_EXAMPLE_INPUTS, _qdq_quantized_adaptive_avg_pool2d, _reference_quantized_adaptive_avg_pool2d, replace_literals_with_new_placeholders, replace_literals_with_new_placeholders),
    (_QUANTIZE_PER_TENSOR_INT8_EXAMPLE_INPUTS, _quantize_per_tensor_int8, _reference_quantize_per_tensor_int8, None, None),
    (_DEQUANTIZE_PER_TENSOR_INT8_EXAMPLE_INPUTS, _dequantize_per_tensor_int8, _reference_dequantize_per_tensor_int8, None, None),
    (_QUANTIZE_PER_CHANNEL_INT8_EXAMPLE_INPUTS, _quantize_per_channel_int8, _reference_quantize_per_channel_int8, _replace_ph_qdq_per_channel_replacement, _replace_ph_qdq_per_channel_replacement),
    (_DEQUANTIZE_PER_CHANNEL_INT8_EXAMPLE_INPUTS, _dequantize_per_channel_int8, _reference_dequantize_per_channel_int8, _replace_ph_qdq_per_channel_replacement, _replace_ph_qdq_per_channel_replacement),
]

def reference_representation_rewrite(model: GraphModule) -> GraphModule:
    remove_tensor_overload_for_qdq_ops(model)
    for example_inputs, pattern, replacement, post_trans_pattern, post_trans_replacement in _EXAMPLE_INPUTS_PATTERN_AND_REPLACEMENTS:
        pattern = get_aten_graph_module(pattern, example_inputs)  # type: ignore[arg-type, assignment]
        remove_tensor_overload_for_qdq_ops(pattern)  # type: ignore[arg-type]
        replacement = get_aten_graph_module(replacement, example_inputs)  # type: ignore[arg-type, assignment]
        remove_tensor_overload_for_qdq_ops(replacement)  # type: ignore[arg-type]
        if post_trans_pattern:
            pattern = post_trans_pattern(pattern)
        if post_trans_replacement:
            replacement = post_trans_replacement(replacement)
        pattern.recompile()  # type: ignore[attr-defined]
        replacement.recompile()  # type: ignore[attr-defined]
        matches = replace_pattern(model, pattern, replacement)
    return model
