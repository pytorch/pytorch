import torch
from torch.ao.quantization._pt2e.quantizer.quantizer import (
    QuantizationConfig,
    QuantizationSpec,
)

def get_act_qspec(quantization_config: QuantizationConfig):
    if quantization_config is None:
        return None
    if quantization_config.activation is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.activation
    assert quantization_spec.qscheme in [
        torch.per_tensor_affine,
        torch.per_tensor_symmetric,
    ]
    if quantization_spec.is_dynamic:
        # TODO: extend this helper function to support dynamic quantization
        raise Exception(
            "Unsupported quantization_spec for activation: {}".format(quantization_spec)
        )
    return quantization_spec

def get_weight_qspec(quantization_config: QuantizationConfig):
    if quantization_config is None:
        return None
    assert quantization_config is not None
    if quantization_config.weight is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.weight
    if quantization_spec.qscheme not in [
        torch.per_tensor_symmetric,
        torch.per_channel_symmetric,
    ]:
        raise ValueError(
            f"Unsupported quantization_spec {quantization_spec} for weight"
        )
    return quantization_spec

def get_bias_qspec(quantization_config: QuantizationConfig):
    if quantization_config is None:
        return None
    assert quantization_config is not None
    if quantization_config.bias is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.bias
    assert (
        quantization_spec.dtype == torch.float
    ), "Only float dtype for bias is supported for bias right now"
    return quantization_spec
