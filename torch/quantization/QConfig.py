from __future__ import absolute_import, division, print_function, unicode_literals
from collections import namedtuple
from .observer import *

default_weight_qoptions = {
    'dtype': torch.qint8,
    'qscheme': torch.per_tensor_affine
}

default_activation_qoptions = {
    'dtype': torch.quint8,
    'qscheme': torch.per_tensor_affine
}

QConfig = namedtuple('QConfig',
                     ['weight', 'activation'])

default_qconfig = QConfig(default_weight_observer(default_weight_qoptions),
                          default_observer(default_activation_qoptions))
