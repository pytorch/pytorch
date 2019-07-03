from __future__ import absolute_import, division, print_function, unicode_literals
from collections import namedtuple
from .observer import *

QConfig = namedtuple('QConfig',
                     ['weight', 'activation'])

default_qconfig = QConfig(default_weight_observer(),
                          default_observer())
