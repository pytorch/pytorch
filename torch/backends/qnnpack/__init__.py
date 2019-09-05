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

class QnnpackModule(types.ModuleType):
    def __init__(self, m, name):
        super(QnnpackModule, self).__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)

    enabled = ContextProp(torch._C._get_qnnpack_enabled, torch._C._set_qnnpack_enabled)

# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = QnnpackModule(sys.modules[__name__], __name__)
