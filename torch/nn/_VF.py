import torch
import sys
import types


class VFModule(types.ModuleType):
    def __getattr__(self, attr):
        return getattr(torch._C._VariableFunctions, attr)

sys.modules[__name__] = VFModule(__name__)
