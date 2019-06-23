from __future__ import absolute_import, division, print_function, unicode_literals
from collections import namedtuple

class QOptions(object):
    def __init__(self, dtype, qscheme):
        super(QOptions, self).__init__()
        self.dtype = dtype
        self.qscheme = qscheme

QConfig = namedtuple('QConfig', ['weight', 'activation'])
