# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/nn/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/nn/quantization/intrinsic/modules/fused.py`.
"""

from torch.ao.nn.quantization.intrinsic import (
    _FusedModule,
    ConvReLU1d,
    ConvReLU2d,
    ConvReLU3d,
    LinearReLU,
    ConvBn1d,
    ConvBn2d,
    ConvBnReLU1d,
    ConvBnReLU2d,
    ConvBn3d,
    ConvBnReLU3d,
    BNReLU2d,
    BNReLU3d,
)
