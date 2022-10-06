# flake8: noqa: F401
r"""Intrinsic QAT Modules

This file is in the process of migration to `torch/ao/nn/intrinsic/qat`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/intrinsic/qat/modules`,
while adding an import statement here.
"""

__all__ = [
    'LinearBn1d',
]

from torch.ao.nn.intrinsic.qat import LinearBn1d
