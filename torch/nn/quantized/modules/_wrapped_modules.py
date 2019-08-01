from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.nn.modules import Module

r"""Base class for all quantized wrapper modules."""
class _BaseWrapperModule(Module):
    __FLOAT_MODULE = None

    def __init__(self):
        super(_BaseWrapperModule, self).__init__()
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.double))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.long))

    def forward(self, *args):
        return self.operation(*args, scale=self.scale,
                              zero_point=self.zero_point)

    @classmethod
    def from_float(cls, mod):
        if __FLOAT_MODULE is None:
            raise NotImplementedError("I don't know what to do with "
                                      + type(mod))
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

r"""Add module wraps torch.ops.quantized.add."""
class Add(_BaseWrapperModule):
    __FLOAT_MODULE = torch.nn.modules.Add

    def __init__(self):
        super(Add, self).__init__()
        self.operation = torch.ops.quantized.add
