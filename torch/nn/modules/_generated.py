
# @generated This file is produced by `torch/quantization/tools/make_module`.

import torch
from torch.nn.modules import Module

r"""Add wraps the torch.add function."""
class Add(Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, *args):
        return torch.add(*args)
