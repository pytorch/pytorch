import torch
from torch.fx import GraphModule
from ..utils import _get_aten_graph_module
from ..utils import _remove_tensor_overload_for_qdq_ops
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.fx.subgraph_rewriter import replace_pattern

__all__ = [
    "reference_representation_rewrite",
]

_QUANTIZED_ADD_EXAMPLE_INPUTS = (
    torch.randn(1, 3, 3, 3).to(torch.int8),
    torch.randn(1).to(torch.float),
    torch.zeros(1).to(torch.int),
    torch.randn(1, 3, 3, 3).to(torch.int8),
    torch.randn(1).to(torch.float),
    torch.zeros(1).to(torch.int),
    torch.randn(1).to(torch.float),
    torch.zeros(1).to(torch.int),
)

def _qdq_quantized_add(x_i8, x_scale, x_zero_point, y_i8, y_scale, y_zero_point, out_scale, out_zero_point):
    quant_min = -128
    quant_max = 127
    x_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(x_i8, x_scale, x_zero_point, quant_min, quant_max, torch.int8)
    y_fp32 = torch.ops.quantized_decomposed.dequantize_per_tensor(y_i8, y_scale, y_zero_point, quant_min, quant_max, torch.int8)
    out_fp32 = x_fp32 + y_fp32
    out_i8 = torch.ops.quantized_decomposed.quantize_per_tensor(
        out_fp32, out_scale, out_zero_point, quant_min, quant_max, torch.int8
    )
    return out_i8

def _reference_quantized_add(x_i8, x_scale, x_zero_point, y_i8, y_scale, y_zero_point, out_scale, out_zero_point):
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

_EXAMPLE_INPUTS_PATTERN_AND_REPLACEMENTS = [
    (_QUANTIZED_ADD_EXAMPLE_INPUTS, _qdq_quantized_add, _reference_quantized_add)
]

def reference_representation_rewrite(model: GraphModule) -> GraphModule:
    _remove_tensor_overload_for_qdq_ops(model)
    for example_inputs, pattern, replacement in _EXAMPLE_INPUTS_PATTERN_AND_REPLACEMENTS:
        pattern_graph = _get_aten_graph_module(pattern, example_inputs)
        _remove_tensor_overload_for_qdq_ops(pattern_graph)
        replacement_graph = _get_aten_graph_module(replacement, example_inputs)
        matches = replace_pattern(model, pattern_graph, replacement_graph)
    return model
