# flake8: noqa: F401
r"""Intrinsic Quantized Modules

This file is in the process of migration to `torch/ao/nn/intrinsic/quantized`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/intrinsic/quantized/modules`,
while adding an import statement here.
"""

from torch.ao.nn.intrinsic.quantized.modules.linear_relu import LinearReLU

__all__ = [
    'LinearReLU',
]
