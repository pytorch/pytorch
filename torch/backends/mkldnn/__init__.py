import sys
import torch
import types
from contextlib import contextmanager


def is_available():
    r"""Returns whether PyTorch is built with MKL-DNN support."""
    return torch._C.has_mkldnn

def set_flags(_enabled):
    orig_flags = (torch._C._get_mkldnn_enabled(),)
    torch._C._set_mkldnn_enabled(_enabled)
    return orig_flags

@contextmanager
def flags(enabled=False):
    orig_flags = set_flags(enabled)
    try:
        yield
    finally:
        set_flags(orig_flags[0])

# from cudnn/__init__.py
#
#
class ContextProp(object):
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter()

    def __set__(self, obj, val):
        self.setter(val)

class MkldnnModule(types.ModuleType):
    def __init__(self, m, name):
        super(MkldnnModule, self).__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)

    enabled = ContextProp(torch._C._get_mkldnn_enabled, torch._C._set_mkldnn_enabled)

# Cool stuff from torch/backends/cudnn/__init__.py and
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = MkldnnModule(sys.modules[__name__], __name__)
