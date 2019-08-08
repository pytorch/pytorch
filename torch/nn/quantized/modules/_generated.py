
# @generated This file is produced by `torch/quantization/tools/make_module`.

import torch
from torch.nn.modules import Module

r"""Add wraps the torch.ops.quantized.add function."""
class Add(Module):
    __FLOAT_MODULE = torch.nn.modules.Add

    def __init__(self):
        super(Add, self).__init__()
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.double))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.long))

    def forward(self, *args):
        return torch.ops.quantized.add(
            *args, scale=self.scale, zero_point=self.zero_point)

    @classmethod
    def from_float(cls, mod):
        assert (hasattr(mod, 'observer')),\
            "Input float module must have observer attached"
        assert (type(mod) == cls.__FLOAT_MODULE),\
            "nnq." + cls.__name__ + ".from_float only works for " \
            + cls.__FLOAT_MODULE.__name__
        scale, zero_point = mod.observer.calculate_qparams()[:2]
        mod = cls()
        mod.scale = torch.tensor(scale, dtype=torch.double)
        mod.zero_point = torch.tensor(zero_point, dtype=torch.long)
        return mod
