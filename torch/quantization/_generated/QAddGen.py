
# This file is generated using `torch.nn.quantization.make_module`
# and saved as `QAddGen.py`.

import torch

r"""AddGen wraps the torch.add function."""
class AddGen(torch.nn.Module):
    def __init__(self, **kwargs):
        super(AddGen, self).__init__()

    def forward(self, *args):
        return self.torch.add(*args)



r"""QAddGen wraps the torch.ops.quantized.add function."""
class QAddGen(torch.nn.Module):
    def __init__(self, **kwargs):
        super(QAddGen, self).__init__()

        self.scale = kwargs.get('scale', 1.0)
        self.zero_point = swargs.get('zero_point', 0)

    def forward(self, *args):
        return self.torch.ops.quantized.add(*args)

    def from_float(self, mod):
        return QAddGen()
