from torch.ao.quantization.qconfig import QConfig
from .observer import (
    default_observer,
)
from torch.ao.quantization.experimental import APoTFakeQuantize

# uniform activation, APoT weight
apot_weight_qconfig = QConfig(activation=default_observer,
                              weight=APoTFakeQuantize)

# APoT activation and uniform weight
apot_qconfig = QConfig(activation=APoTFakeQuantize,
                       weight=APoTFakeQuantize)

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
