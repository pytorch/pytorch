# flake8: noqa: F401
r"""Quantized Reference Modules.

This module is in the process of migration to
`torch/ao/nn/quantized/reference`, and is kept here for
compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/reference`,
while adding an import statement here.
"""

from torch.ao.nn.quantized.reference.modules.conv import (
    _ConvNd,
    _ConvTransposeNd,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
