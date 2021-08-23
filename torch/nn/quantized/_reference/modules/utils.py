import torch

def _quantize_and_dequantize_weight(weight, weight_qscheme, weight_dtype, weight_scale, weight_zero_point, weight_axis):
    """ Quantize and then dequantize the weight based on
    the quantization parameters
    """
    if weight_qscheme == torch.per_tensor_affine:
        weight = torch.quantize_per_tensor(weight, weight_scale, weight_zero_point, weight_dtype)
        weight_dequant = weight.dequantize()
    elif weight_qscheme == torch.per_channel_affine:
        weight = torch.quantize_per_channel(
            weight, weight_scale,
            weight_zero_point, weight_axis.item(), weight_dtype)
        weight_dequant = weight.dequantize()
    else:
        weight_dequant = weight
    return weight_dequant
