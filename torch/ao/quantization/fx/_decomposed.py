import torch
from torch.library import Library, impl

quantized_decomposed_lib = Library("quantized_decomposed", "DEF")

quantized_decomposed_lib.define(
    "quantize_per_tensor(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "quantize_per_tensor", "CompositeExplicitAutograd")
def quantize_per_tensor(input, scale, zero_point, quant_min, quant_max, dtype):
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    quant_min_lower_bound = 0
    quant_max_upper_bound = 0
    if dtype == torch.uint8:
        quant_min_lower_bound = 0
        quant_max_upper_bound = 255
    elif dtype == torch.int8:
        quant_min_lower_bound = -128
        quant_max_upper_bound = 127
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    assert quant_min >= quant_min_lower_bound, \
        "quant_min out of bound for dtype, " \
        f"quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}"

    assert quant_max <= quant_max_upper_bound, \
        "quant_max out of bound for dtype, " \
        f"quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}"

    inv_scale = 1.0 / scale
    return torch.clamp(torch.round(input * inv_scale) + zero_point, quant_min, quant_max).to(dtype)

quantized_decomposed_lib.define(
    "dequantize_per_tensor(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "dequantize_per_tensor", "CompositeExplicitAutograd")
def dequantize_per_tensor(input, scale, zero_point, quant_min, quant_max, dtype):
    assert input.dtype == dtype, f"Expecting input to have dtype: {dtype}"
    if dtype in [torch.uint8, torch.int8]:
        return (input.to(torch.float32) - zero_point) * scale
    else:
        raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")
