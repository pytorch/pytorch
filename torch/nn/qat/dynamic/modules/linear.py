# flake8: noqa: F401
r"""QAT Modules

This file is in the process of migration to `torch/ao/nn/qat/dynamic`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/qat/dynamic/modules`,
while adding an import statement here.
"""
from torch.ao.nn.qat.dynamic.modules.linear import Linear
