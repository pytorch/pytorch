
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.nn._intrinsic import AddReLU as NNAddReLU

class AddReLU(torch.nn.Module):
    _FLOAT_MODULE = NNAddReLU

    def __init__(self):
        super(AddReLU, self).__init__()

    def forward(self, a, b):
        return torch.ops.quantized.add_relu(a, b)
