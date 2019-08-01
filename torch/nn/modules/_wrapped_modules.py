from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.nn.modules import Module

r"""Base class for all wrapper modules."""
class _BaseWrapperModule(Module):
    def __init__(self):
        super(_BaseWrapperModule, self).__init__()
        self.operation = None

    def forward(self, *args):
        return self.operation(*args)

r"""Add module wraps torch.add."""
class Add(_BaseWrapperModule):
    def __init__(self):
        super(Add, self).__init__()
        self.operation = torch.ops.quantized.add
