from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import (
    default_observer
)
import torch
from torch.ao.quantization.experimental.fake_quantize import APoTFakeQuantize

# uniform activation and weight
uniform_qconfig = QConfig(activation=default_observer.with_args(quant_min=0, quant_max=255, dtype=torch.quint8),
                          weight=default_observer.with_args(quant_min=0, quant_max=255, dtype=torch.qint8))

# uniform activation, APoT weight
apot_weight_qconfig = QConfig(activation=default_observer.with_args(dtype=torch.quint8),
                              weight=APoTFakeQuantize.with_args(b=8, k=2, dtype=torch.qint8))

# APoT activation and uniform weight
apot_qconfig = QConfig(activation=APoTFakeQuantize.with_args(b=8, k=2, dtype=torch.quint8),
                       weight=APoTFakeQuantize.with_args(b=8, k=2, dtype=torch.qint8))

def get_uniform_qconfig():
    """
    Returns the uniform qconfig
    """
    return uniform_qconfig

def get_apot_weights_qconfig():
    """
    Returns qconfig with uniform activation,
    APoT weight
    """
    return apot_weight_qconfig

def get_apot_qconfig():
    """
    Returns the APoT qconfig
    """
    return apot_qconfig
