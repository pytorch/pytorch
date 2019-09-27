from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import torch
import types

# This function should correspond to the enums present in c10/core/QEngine.h
def get_qengine_id(qengine):
    # type: (str) -> int
    if qengine == 'none':
        ret = 0
    elif qengine == 'fbgemm':
        ret = 1
    elif qengine == 'qnnpack':
        ret = 2
    else:
        ret = -1
        raise RuntimeError("{} is not a valid value for quantized engine".format(qengine))
    return ret

# This function should correspond to the enums present in c10/core/QEngine.h
def get_qengine_str(qengine):
    # type: (int) -> str
    all_engines = {0 : 'none', 1 : 'fbgemm', 2 : 'qnnpack'}
    return all_engines.get(qengine)

def get_supported_qengines():
    qengines = torch._C._supported_qengines()
    return [get_qengine_str(qe) for qe in qengines]

class ContextProp(object):
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return get_qengine_str(self.getter())

    def __set__(self, obj, val):
        self.setter(get_qengine_id(val))

class QuantizedEngine(types.ModuleType):
    def __init__(self, m, name):
        super(QuantizedEngine, self).__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)
    engine = ContextProp(torch._C._get_qengine, torch._C._set_qengine)

# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = QuantizedEngine(sys.modules[__name__], __name__)
