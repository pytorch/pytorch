import torch
from ._device_daemon import daemon

# Create our python implementation dict so that the C++ module
# can access it during its initialization
_IMPL_REGISTRY = {}

# Load the C++ Module
import pytorch_openreg._C

# Define all the implementations in the registry
def register(fn):
    _IMPL_REGISTRY[fn.__name__[1:]] = fn
    return fn

def register_same_name(name):
    def _(*args, **kwargs):
        return daemon.exec(name, *args, **kwargs)
    _IMPL_REGISTRY[name] = _

register_same_name("deviceCount")
register_same_name("getDevice")
register_same_name("uncheckedSetDevice")
register_same_name("malloc")
register_same_name("free")

# Module used for our backend
class _OpenRegMod():
    pass

# Set all the appropriate state on PyTorch
torch.utils.rename_privateuse1_backend("openreg")
torch._register_device_module("openreg", _OpenRegMod())

_openreg_lib = torch.library.Library("_", "IMPL")

def _openreg_kernel_fallback(op, *args, **kwargs):
    print(op)
    assert op is torch.ops.aten.empty.memory_format
    # FIXME: this returns a cpu Tensor which is NOT ok.

    return torch.empty(args[0])

_openreg_lib.fallback(_openreg_kernel_fallback, dispatch_key="PrivateUse1")


