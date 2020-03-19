from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import torch
import types

class _XNNPACKEnabled(object):
    def __get__(self, obj, objtype):
        return torch._C._is_xnnpack_enabled()

    def __set__(self, obj, val):
        raise RuntimeError("Assignment not supported")

class XNNPACKEngine(types.ModuleType):
    def __init__(self, m, name):
        super(XNNPACKEngine, self).__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)

    enabled = _XNNPACKEnabled()

# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = XNNPACKEngine(sys.modules[__name__], __name__)
