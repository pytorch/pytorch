"""
This makes the functions in torch._C._VariableFunctions available as
    torch._VF.<funcname>
without mypy being able to find them.

A subset of those functions are mapped to ATen functions in
torch/jit/_builtins.py

See https://github.com/pytorch/pytorch/issues/21478 for the reason for
introducing torch._VF

"""
import torch
import sys
import types


class VFModule(types.ModuleType):
    vf: types.ModuleType

    def __init__(self, name):
        super(VFModule, self).__init__(name)
        self.vf = torch._C._VariableFunctions

    def __getattr__(self, attr):
        return getattr(self.vf, attr)


sys.modules[__name__] = VFModule(__name__)
