import torch


# Global properties of our device
NUM_DEVICES = 7

# Create our python implementation dict so that the C++ module
# can access it during its initialization
_IMPL_REGISTRY = {}

# Load the C++ Module
import pytorch_openreg._C  # noqa: F401


# Define all the implementations in the registry
def register(fn):
    _IMPL_REGISTRY[fn.__name__[1:]] = fn
    return fn


@register
def _deviceCount():
    return NUM_DEVICES


# Module used for our backend
class _OpenRegMod:
    pass


# Set all the appropriate state on PyTorch
torch.utils.rename_privateuse1_backend("openreg")
torch._register_device_module("openreg", _OpenRegMod())

_openreg_lib = torch.library.Library("_", "IMPL")  # ignore TOR901


def _openreg_kernel_fallback(op, *args, **kwargs):
    print("Calling ", op)
    assert op is torch.ops.aten.empty.memory_format
    # FIXME: this returns a cpu Tensor which is NOT ok.
    return torch.empty(args[0])


_openreg_lib.fallback(_openreg_kernel_fallback, dispatch_key="PrivateUse1")
