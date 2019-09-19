from __future__ import absolute_import, division, print_function, unicode_literals
from collections import namedtuple
from .observer import *
from .fake_quantize import *

QConfig = namedtuple('QConfig',
                     ['activation', 'weight'])

default_qconfig = QConfig(activation=default_observer(),
                          weight=default_weight_observer())

default_debug_qconfig = QConfig(weight=default_weight_observer(),
                                activation=default_debug_observer())

QConfig_dynamic = namedtuple('QConfig_dynamic', ['weight'])

default_dynamic_qconfig = QConfig_dynamic(weight=default_weight_observer())

default_qat_qconfig = QConfig(activation=default_fake_quant(),
                              weight=default_weight_fake_quant())
