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

if torch.distributed.is_available():
    try:
        torch.distributed.Backend.register_backend(
            "occl", torch_openreg._C._createProcessGroupOCCL, devices=["openreg"]
        )
    except Exception as e:
        raise RuntimeError("Failed to register 'occl' process group backend.") from e


# LITERALINCLUDE START: AUTOLOAD
def _autoload():
    # It is a placeholder function here to be registered as an entry point.
    pass


# LITERALINCLUDE END: AUTOLOAD
