# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/fuse_modules.py`, while adding an import statement
here.
"""

# TODO: These functions are not used outside the `fuse_modules.py`
#       Keeping here for now, need to remove them later.
from torch.ao.quantization.fuse_modules import (
    _fuse_modules,
    _get_module,
    _set_module,
    fuse_known_modules,
    fuse_modules,
    get_fuser_method,
)

# for backward compatibility
from torch.ao.quantization.fuser_method_mappings import fuse_conv_bn, fuse_conv_bn_relu
