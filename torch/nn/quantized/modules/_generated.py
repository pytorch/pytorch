
# @generated This file is produced by `torch/quantization/tools/make_module`.

import torch

r"""Add wraps the torch.ops.quantized.add function."""
class Add(torch.nn.Module):
    __FLOAT_MODULE = torch.nn.modules.Add

    def __init__(self, scale=1.0, zero_point=0):
        super(Add, self).__init__()
        self.scale = 1.0
        self.zero_point = 0

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
        return Add(
            scale=qparams[0].item(), zero_point=qparams[1].item())
