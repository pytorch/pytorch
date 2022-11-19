# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/fuser_method_mappings.py`, while adding an import statement
here.
"""
from torch.ao.quantization.fuser_method_mappings import (
    fuse_conv_bn,
    fuse_conv_bn_relu,
    fuse_linear_bn,
    _DEFAULT_OP_LIST_TO_FUSER_METHOD,
    get_fuser_method,
)
