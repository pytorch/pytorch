import torch

# Create our python implementation dict so that the C++ module
# can access it during its initialization
# Also register aten impls
from ._aten_impl import _IMPL_REGISTRY as _IMPL_REGISTRY  # noqa: F401
from ._device_daemon import NUM_DEVICES as NUM_DEVICES


# Load the C++ Module
import pytorch_openreg._C  # noqa: F401  # usort: skip


""" TODO Add back and rename all 'foo' to 'openreg'


# Module used for our backend
class _OpenRegMod:
    pass


# Set all the appropriate state on PyTorch
torch.utils.rename_privateuse1_backend("openreg")
torch._register_device_module("openreg", _OpenRegMod())
"""
