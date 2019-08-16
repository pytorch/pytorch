
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.nn.quantized.modules._base_modules import _BaseQuantizedModule


class AddReLU(torch.nn.Module, _BaseQuantizedModule):
    _FLOAT_MODULE = torch.nn._intrinsic.AddReLU

    def __init__(self):
        super(AddReLU, self).__init__()

    def forward(self, a, b):
        return torch.ops.quantized.add_relu(a, b, scale=self.scale,
                                            zero_point=self.zero_point)
