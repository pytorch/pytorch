from __future__ import absolute_import, division, print_function, unicode_literals
from collections import namedtuple
from .observer import *
from .fake_quantize import *

QConfig = namedtuple('QConfig',
                     ['weight', 'activation'])

default_qconfig = QConfig(default_weight_observer(),
                          default_observer())

QConfig_dynamic = namedtuple('QConfig_dynamic', ['weight'])

default_dynamic_qconfig = QConfig_dynamic(default_weight_observer())

default_qat_qconfig = QConfig(default_weight_fake_quant(),
                              default_fake_quant())
