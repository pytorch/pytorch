import torch
from torch.fx import GraphModule
from ..utils import get_aten_graph_module
from ..utils import remove_tensor_overload_for_qdq_ops
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.fx.subgraph_rewriter import replace_pattern
from torch._higher_order_ops.out_dtype import out_dtype

__all__ = [
    "reference_representation_rewrite",
]

_QUANTIZED_ADD_EXAMPLE_INPUTS = (
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

def _qdq_quantized_add_relu(x_i8, x_scale, x_zero_point, y_i8, y_scale, y_zero_point, out_scale, out_zero_point, quant_min, quant_max):
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
    # See comments for `_reference_quantized_add` for more information on how to derive the formula for out_i8 based on x_i8 and y_i8
    """
    x_i32 = x_i8.to(torch.int32)
    y_i32 = y_i8.to(torch.int32)
    # TODO: use out_dtype op
    # x_i32 = out_dtype(torch.ops.aten.mul, torch.int32, (x_i32 - x_zero_point), (x_scale / out_scale))
    # y_i32 = out_dtype(torch.ops.aten.mul, torch.int32, (y_i32 - y_zero_point), (y_scale / out_scale))
    x_i32 = torch.round((x_scale / out_scale) * (x_i32 - x_zero_point)).to(torch.int32)
    y_i32 = torch.round((y_scale / out_scale) * (y_i32 - y_zero_point)).to(torch.int32)
    out_i32 = x_i32 + y_i32 + out_zero_point
    out_i32 = torch.ops.aten.clamp(out_i32, out_zero_point)
    quant_min = -128
    quant_max = 127
    out_i8 = torch.ops.aten.clamp(out_i32, quant_min, quant_max).to(torch.int8)
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
    torch.randint(-128, 127, (1, 3, 3, 3)).to(dtype=torch.int8),
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

_EXAMPLE_INPUTS_PATTERN_AND_REPLACEMENTS = [
    (_QUANTIZED_ADD_EXAMPLE_INPUTS, _qdq_quantized_add_relu, _reference_quantized_add_relu),
    (_QUANTIZED_ADD_EXAMPLE_INPUTS, _qdq_quantized_add, _reference_quantized_add),
    (_QUANTIZE_PER_TENSOR_INT8_EXAMPLE_INPUTS, _quantize_per_tensor_int8, _reference_quantize_per_tensor_int8),
    (_DEQUANTIZE_PER_TENSOR_INT8_EXAMPLE_INPUTS, _dequantize_per_tensor_int8, _reference_dequantize_per_tensor_int8),
]

def reference_representation_rewrite(model: GraphModule) -> GraphModule:
    remove_tensor_overload_for_qdq_ops(model)
    for example_inputs, pattern, replacement in _EXAMPLE_INPUTS_PATTERN_AND_REPLACEMENTS:
        pattern = get_aten_graph_module(pattern, example_inputs)  # type: ignore[arg-type, assignment]
        remove_tensor_overload_for_qdq_ops(pattern)  # type: ignore[arg-type]
        replacement = get_aten_graph_module(replacement, example_inputs)  # type: ignore[arg-type, assignment]
        remove_tensor_overload_for_qdq_ops(replacement)  # type: ignore[arg-type]
        matches = replace_pattern(model, pattern, replacement)
    return model
