# flake8: noqa: F401
r"""Quantized Dynamic Modules.

This file is in the process of migration to `torch/ao/nn/quantized/dynamic`,
and is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/dynamic/modules`,
while adding an import statement here.
"""

__all__ = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']

from torch.ao.nn.quantized.dynamic.modules.conv import Conv1d
from torch.ao.nn.quantized.dynamic.modules.conv import Conv2d
from torch.ao.nn.quantized.dynamic.modules.conv import Conv3d
from torch.ao.nn.quantized.dynamic.modules.conv import ConvTranspose1d
from torch.ao.nn.quantized.dynamic.modules.conv import ConvTranspose2d
from torch.ao.nn.quantized.dynamic.modules.conv import ConvTranspose3d
