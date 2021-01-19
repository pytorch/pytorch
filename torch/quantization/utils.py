"""
Utils shared by different modes of quantization (eager/graph)
"""
import torch
from .quant_type import QuantType, quant_type_to_str

def get_combined_dict(default_dict, additional_dict):
    d = default_dict.copy()
    d.update(additional_dict)
    return d

def is_per_tensor(qscheme):
    return qscheme == torch.per_tensor_affine or \
        qscheme == torch.per_tensor_symmetric

def is_per_channel(qscheme):
    return qscheme in [torch.per_channel_affine,
                       torch.per_channel_affine_float_qparams,
                       torch.per_channel_symmetric]

def get_swapped_custom_module_class(custom_module, custom_module_class_mapping, qconfig):
    """ Get the observed/quantized custom module class that we need
    to swap `custom_module` to
    Input:
        custom_module: input, can be an instance of either a float or observed custom module
        custom_module_class_mapping: the float to observed or observed to quantized custom module class mapping
        qconfig: qconfig configured for the custom module

    Output:
        corresponding observed/quantized custom module class for input custom module instance
    """
    quant_type = get_quant_type(qconfig)
    quant_type_str = quant_type_to_str(quant_type)
    class_mapping = custom_module_class_mapping.get(quant_type_str, {})
    assert type(custom_module) in class_mapping, "did not find corresponding observed " \
        "module class for {} in mapping: {}".format(type(custom_module), class_mapping)
    return class_mapping[type(custom_module)]

def activation_is_statically_quantized(qconfig):
    """ Given a qconfig, decide if the activation needs to be
    statically quantized or not
    """
    assert qconfig is not None
    activation = qconfig.activation()
    return activation.dtype in [torch.quint8, torch.qint8]

def weight_dtype(qconfig):
    assert qconfig is not None
    weight = qconfig.weight()
    return weight.dtype

def weight_is_statically_quantized(qconfig):
    """ Given a qconfig, decide if the weight needs to be
    quantized or not
    """
    return weight_dtype(qconfig) in [torch.quint8, torch.qint8]

def get_qconfig_dtypes(qconfig):
    r""" returns the qconfig tuple for qconfig:
    (activation_dtype, weight_dtype, activation_compute_dtype)
    """
    assert qconfig is not None
    activation = qconfig.activation()
    weight = qconfig.weight()
    compute_dtype = activation.compute_dtype if hasattr(activation, 'compute_dtype') else None
    return (activation.dtype, weight.dtype, compute_dtype)

def get_quant_type(qconfig):
    assert qconfig is not None
    activation = qconfig.activation()
    weight = qconfig.weight()
    static_dtypes = [torch.quint8, torch.qint8]
    if weight.dtype in static_dtypes:
        if activation.dtype in static_dtypes:
            return QuantType.STATIC
        elif hasattr(activation, 'compute_dtype') and activation.compute_dtype in static_dtypes:
            return QuantType.DYNAMIC
        else:
            return QuantType.WEIGHT_ONLY

    if weight.dtype == torch.float16:
        if activation.dtype == torch.float:
            return QuantType.DYNAMIC

    raise Exception("Unrecognized dtype combination in get_quant_type: activation({}),"
                    "weight({})".format(activation.dtype, weight.dtype))
