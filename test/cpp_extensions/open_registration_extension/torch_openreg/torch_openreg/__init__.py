import sys

import torch


if sys.platform == "win32":
    from ._utils import _load_dll_libraries

    _load_dll_libraries()
    del _load_dll_libraries

import torch_openreg._C  # type: ignore[misc]
import torch_openreg.openreg


torch.utils.rename_privateuse1_backend("openreg")
torch._register_device_module("openreg", torch_openreg.openreg)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

# Register gloo as the default distributed backend for openreg
# This is needed for distributed tests to work with the openreg device
import torch.distributed as dist


if hasattr(dist, "Backend") and hasattr(dist.Backend, "default_device_backend_map"):
    dist.Backend.default_device_backend_map["openreg"] = "gloo"


# LITERALINCLUDE START: AUTOLOAD
def _autoload():
    # It is a placeholder function here to be registered as an entry point.
    pass


# LITERALINCLUDE END: AUTOLOAD
