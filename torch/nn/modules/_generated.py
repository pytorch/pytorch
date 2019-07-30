
# @generated This file is produced by `torch/quantization/tools/make_module`.

import torch

r"""Add wraps the torch.add function."""
class Add(torch.nn.Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, *args):
        return torch.add(*args)
