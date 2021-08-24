import torch

def _init_weight_qparams(module, weight_qparams):
    if weight_qparams is None:
        weight_qparams = {
            "qscheme": torch.per_tensor_affine,
            "dtype": torch.quint8,
            "scale": 1.0,
            "zero_point": 0
        }
    module.weight_qscheme = weight_qparams["qscheme"]
    module.weight_dtype = weight_qparams["dtype"]
    assert module.weight_qscheme in [None, torch.per_tensor_affine, torch.per_channel_affine], \
    Exception(f"qscheme: {module.weight_qscheme} is not support in reference quantized linear module")
    if module.weight_qscheme is not None:
        module.register_buffer("weight_scale", torch.tensor(weight_qparams["scale"]))
        module.register_buffer("weight_zero_point", torch.tensor(weight_qparams["zero_point"]))
        if module.weight_qscheme == torch.per_channel_affine:
            module.register_buffer("weight_axis", torch.tensor(weight_qparams["axis"]))
        else:
            # added for TorchScriptability, not used
            module.register_buffer("weight_axis", torch.tensor(0))


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

def _save_weight_qparams(destination, prefix, weight_qscheme, weight_dtype, weight_scale, weight_zero_point, weight_axis):
    destination[prefix + "weight_qscheme"] = weight_qscheme
    destination[prefix + "weight_dtype"] = weight_dtype
    if weight_qscheme is not None:
        destination[prefix + "weight_scale"] = weight_scale
        destination[prefix + "weight_zero_point"] = weight_zero_point
        if weight_qscheme == torch.per_channel_affine:
            destination[prefix + "weight_axis"] = weight_axis

def _load_weight_qparams(module, state_dict):
    module.weight_qscheme = state_dict[prefix + "weight_qscheme"]
    module.weight_dtype = state_dict[prefix + "weight_dtype"]
    state_dict.pop(prefix + "weight_qscheme")
    state_dict.pop(prefix + "weight_dtype")
    if module.weight_qscheme is not None:
        module.weight_scale = state_dict[prefix + "weight_scale"]
        module.weight_zero_point = state_dict[prefix + "weight_zero_point"]
        state_dict.pop(prefix + "weight_scale")
        state_dict.pop(prefix + "weight_zero_point")
        if module.weight_qscheme == torch.quantize_per_channel:
            module.weight_axis = state_dict[prefix + "weight_axis"]
