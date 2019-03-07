import torch
import sys
import types


class VFModule(types.ModuleType):
    def __init__(self, name):
        super(VFModule, self).__init__(name)
        self.vf = torch._C._VariableFunctions

    def __getattr__(self, attr):
        return getattr(self.vf, attr)

sys.modules[__name__] = VFModule(__name__)
