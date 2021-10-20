# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/nn/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/nn/quantization/intrinsic/quantized/modules/conv_relu.py`.
"""

from torch.ao.nn.quantization.intrinsic.quantized.modules.conv_relu import (
    ConvReLU1d,
    ConvReLU2d,
    ConvReLU3d,
)
