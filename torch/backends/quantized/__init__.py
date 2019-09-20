from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import torch
import types

class ContextProp(object):
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter()

    def __set__(self, obj, val):
        self.setter(val)

class QuantizedEngine(types.ModuleType):
    def __init__(self, m, name):
        super(QuantizedEngine, self).__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)
    # TODO: replace with strings(https://github.com/pytorch/pytorch/pull/26330/files#r324951460)
    engine = ContextProp(torch._C._get_qengine, torch._C._set_qengine)

# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = QuantizedEngine(sys.modules[__name__], __name__)
