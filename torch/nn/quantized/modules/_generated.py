
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
            "nnq.Add.from_float only works for " + cls.__FLOAT_MODULE.__name__
        qparams = mod.observer.calculate_qparams()
        mod = Add(scale=qparams[0].item(), zero_point=qparams[1].item())
        mod.scale = torch.tensor(qparams[0], dtype=torch.double)
        mod.zero_point = torch.tensor(qparams[1], dtype=torch.long)
        return mod
