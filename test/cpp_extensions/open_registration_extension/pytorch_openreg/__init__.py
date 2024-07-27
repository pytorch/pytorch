# Load the C++ Module
import torch

# Create our python implementation dict so that the C++ module
# can access it during its initialization
# Also register aten impls
from ._aten_impl import _IMPL_REGISTRY as _IMPL_REGISTRY

import pytorch_openreg._C

# Module used for our backend
class _OpenRegMod:
    pass


# Set all the appropriate state on PyTorch
torch.utils.rename_privateuse1_backend("openreg")
torch._register_device_module("openreg", _OpenRegMod())
