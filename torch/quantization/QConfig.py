from __future__ import absolute_import, division, print_function, unicode_literals

class QConfig(object):
    def __init__(self, q_dtype, q_scheme):
        super(QConfig, self).__init__()
        self.q_dtype = q_dtype
        self.q_scheme = q_scheme
