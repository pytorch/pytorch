# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/nn/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/nn/quantization/intrinsic/qat/modules/conv_fused.py`.
"""

from torch.ao.nn.quantization.intrinsic.qat.modules import (
    _BN_CLASS_MAP,
    MOD,
    _ConvBnNd,
    ConvBn1d,
    ConvBn2d,
    ConvBnReLU2d,
    ConvReLU2d,
    ConvBn3d,
    ConvBnReLU3d,
    ConvReLU3d,
    update_bn_stats,
    freeze_bn_stats,
)
