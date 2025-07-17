import sys
import os

import torch


if sys.platform == "win32":
    from ._utils import _load_dll_libraries

    _load_dll_libraries()
    del _load_dll_libraries


import torch_openreg._C  # type: ignore[misc]
import torch_openreg.openreg


def _autoload():
    os.environ["IS_CUSTOM_DEVICE_BACKEND_IMPORTED"] = "1"

    torch.utils.rename_privateuse1_backend("openreg")
    torch._register_device_module("openreg", torch_openreg.openreg)
    torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
