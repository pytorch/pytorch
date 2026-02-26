r"""
This package introduces support for the TPU backend.

The TPU support is mainly provided by the torch_tpu package.
"""

try:
    from torch_tpu import api as tpu_api  # type: ignore[import]
except ImportError:
    tpu_api = None

def current_device() -> int:
    r"""Return the index of a currently selected device."""
    if tpu_api is None:
        return 0
    return tpu_api._device_module._DeviceModule.current_device()
