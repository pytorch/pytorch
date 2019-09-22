from __future__ import absolute_import, division, print_function, unicode_literals
from collections import namedtuple
from .observer import *
from .fake_quantize import *

QConfig = namedtuple('QConfig',
                     ['activation', 'weight'])

def get_default_qconfig(backend='fbgemm'):
    if backend == 'fbgemm':
        qconfig = QConfig(activation=default_l2_observer(),
                          weight=default_per_channel_weight_observer())
    else:
        qconfig = QConfig(activation=default_l2_observer(reduce_range=False),
                          weight=default_weight_observer())
    return qconfig

def get_default_qat_qconfig(backend='fbgemm'):
    if backend == 'fbgemm':
        qconfig = QConfig(activation=l2_fake_quant(),
                          weight=default_per_channel_weight_fake_quant())
    else:
        qconfig = QConfig(activation=l2_fake_quant(reduce_range=False),
                          weight=default_weight_fake_quant())
    return qconfig




default_qconfig = QConfig(activation=default_observer(),
                          weight=default_weight_observer())
default_per_channel_qconfig = QConfig(activation=default_observer(),
                                      weight=default_per_channel_weight_observer())
default_debug_qconfig = QConfig(weight=default_weight_observer(),
                                activation=default_debug_observer())

QConfig_dynamic = namedtuple('QConfig_dynamic', ['weight'])

default_dynamic_qconfig = QConfig_dynamic(weight=default_weight_observer())

default_qat_qconfig = QConfig(activation=default_fake_quant(),
                              weight=default_weight_fake_quant())
# Configs for simulating weight only quantization for debugging
# Use only to analyze accuracy tradeoffs
default_weight_only_quant_qconfig = QConfig(activation=observer(torch.nn.Identity), weight=default_weight_fake_quant())
default_activation_only_quant_qconfig = QConfig(activation=default_fake_quant(), weight=observer(torch.nn.Identity))
