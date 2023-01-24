# flake8: noqa: F401
r"""Intrinsic QAT Modules

This file is in the process of migration to `torch/ao/nn/intrinsic/qat`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/intrinsic/qat/modules`,
while adding an import statement here.
"""

__all__ = [
    # Modules
    'ConvBn1d',
    'ConvBnReLU1d',
    'ConvReLU1d',
    'ConvBn2d',
    'ConvBnReLU2d',
    'ConvReLU2d',
    'ConvBn3d',
    'ConvBnReLU3d',
    'ConvReLU3d',
    # Utilities
    'freeze_bn_stats',
    'update_bn_stats',
]

from torch.ao.nn.intrinsic.qat import ConvBn1d
from torch.ao.nn.intrinsic.qat import ConvBnReLU1d
from torch.ao.nn.intrinsic.qat import ConvReLU1d
from torch.ao.nn.intrinsic.qat import ConvBn2d
from torch.ao.nn.intrinsic.qat import ConvBnReLU2d
from torch.ao.nn.intrinsic.qat import ConvReLU2d
from torch.ao.nn.intrinsic.qat import ConvBn3d
from torch.ao.nn.intrinsic.qat import ConvBnReLU3d
from torch.ao.nn.intrinsic.qat import ConvReLU3d
from torch.ao.nn.intrinsic.qat import freeze_bn_stats
from torch.ao.nn.intrinsic.qat import update_bn_stats
