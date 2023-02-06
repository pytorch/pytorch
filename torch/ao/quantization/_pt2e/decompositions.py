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
