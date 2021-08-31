import torch
from typing import Dict, Any

def _quantize_and_dequantize_weight(
        weight: torch.Tensor,
        weight_qscheme: torch.qscheme,
        weight_dtype: torch.dtype,
        weight_scale: torch.Tensor,
        weight_zero_point: torch.Tensor,
        weight_axis: torch.Tensor):
    """ Quantize and then dequantize the weight based on
    the quantization parameters
    """
    if weight_qscheme == torch.per_tensor_affine:
        weight = torch.quantize_per_tensor(weight, weight_scale, weight_zero_point, weight_dtype)
        weight_dequant = weight.dequantize()
    elif weight_qscheme == torch.per_channel_affine:
        weight = torch.quantize_per_channel(
            weight, weight_scale,
            weight_zero_point, weight_axis.item(), weight_dtype)  # type: ignore[arg-type]
        weight_dequant = weight.dequantize()
    else:
        weight_dequant = weight
    return weight_dequant

def _save_weight_qparams(destination, prefix, weight_qscheme, weight_dtype, weight_scale, weight_zero_point, weight_axis):
    destination[prefix + "weight_qscheme"] = weight_qscheme
    destination[prefix + "weight_dtype"] = weight_dtype
    if weight_qscheme is not None:
        destination[prefix + "weight_scale"] = weight_scale
        destination[prefix + "weight_zero_point"] = weight_zero_point
        if weight_qscheme == torch.per_channel_affine:
            destination[prefix + "weight_axis"] = weight_axis

def _get_weight_qparam_keys(
        state_dict: Dict[str, Any],
        prefix: str):
    keys = ["weight_qscheme", "weight_dtype"]
    weight_qscheme = state_dict[prefix + "weight_qscheme"]
    if weight_qscheme is not None:
        keys.append("weight_scale")
        keys.append("weight_zero_point")
        if weight_qscheme == torch.quantize_per_channel:
            keys.append("weight_axis")
    return keys
