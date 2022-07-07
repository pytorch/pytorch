from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.observer import (
    default_observer
)
import torch

import sys
from torch.ao.quantization.experimental.observer import APoTObserver
from torch.ao.quantization.experimental.fake_quantize import APoTFakeQuantize

# uniform activation and weight
uniform_qconfig = QConfig(activation=default_observer.with_args(dtype=torch.qint8),
                          weight=default_observer.with_args(dtype=torch.qint8))

# uniform activation, APoT weight
apot_weight_qconfig = QConfig(activation=default_observer,
                              weight=APoTFakeQuantize.with_args(b=4, k=2))

# APoT activation and uniform weight
apot_qconfig = QConfig(activation=APoTFakeQuantize.with_args(b=6, k=3),
                       weight=APoTFakeQuantize.with_args(b=4, k=2))

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
