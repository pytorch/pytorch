# TODO: move _decomposed.py file to _pt2e folder
import torch
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch._decomp import register_decomposition

@register_decomposition(torch.ops.quantized_decomposed.quantize_per_tensor)
def quantize_per_tensor(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    inv_scale = 1.0 / scale
    return torch.clamp(
        torch.round(input * inv_scale) + zero_point, quant_min, quant_max
    ).to(dtype)

@register_decomposition(torch.ops.quantized_decomposed.dequantize_per_tensor)
def dequantize_per_tensor(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return (input.to(torch.float32) - zero_point) * scale

# Helper function used to implement per-channel quantization against any axis
def _permute_to_axis_zero(x, axis):
    new_axis_list = list(range(x.dim()))
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = x.permute(tuple(new_axis_list))
    return y, new_axis_list

@register_decomposition(torch.ops.quantized_decomposed.quantize_per_channel)
def quantize_per_channel(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        axis: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    input, permute_axis_list = _permute_to_axis_zero(input, axis)
    res = torch.zeros_like(input)

    for i in range(input.size(0)):
        res[i] = torch.clamp(
            torch.round(input[i] * (1.0 / scales[i])) + zero_points[i],
            quant_min,
            quant_max
        )

    out = res.permute(tuple(permute_axis_list))
    return out.to(dtype)

@register_decomposition(torch.ops.quantized_decomposed.dequantize_per_channel)
def dequantize_per_channel(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        axis: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    input, permute_axis_list = _permute_to_axis_zero(input, axis)
    res = torch.zeros_like(input, dtype=torch.float32)

    for i in range(input.size(0)):
        # TODO: investigate why
        # (input[i] - zero_points[i]).to(torch.float32) * scales[i]
        # failed the test
        res[i] = (input[i].to(torch.float32) - zero_points[i]) * scales[i]

    out = res.permute(tuple(permute_axis_list))
    return out
