# flake8: noqa: F401
r"""Quantized Modules

This file is in the process of migration to `torch/ao/nn/quantized`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/modules`,
while adding an import statement here.
"""

from torch.ao.nn.quantized.modules.activation import ELU
from torch.ao.nn.quantized.modules.activation import Hardswish
from torch.ao.nn.quantized.modules.activation import LeakyReLU
from torch.ao.nn.quantized.modules.activation import MultiheadAttention
from torch.ao.nn.quantized.modules.activation import PReLU
from torch.ao.nn.quantized.modules.activation import ReLU6
from torch.ao.nn.quantized.modules.activation import Sigmoid
from torch.ao.nn.quantized.modules.activation import Softmax
