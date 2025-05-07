import types

import torch

# Create our python implementation dict so that the C++ module
# can access it during its initialization and also register aten impls.
from ._aten_impl import impl_factory as impl_factory  # noqa: F401
from ._device_daemon import driver


# Load the C++ Module
import pytorch_openreg._C  # isort:skip # type: ignore[import] # noqa: F401


def _create_module():
    module = types.ModuleType("_OpenRegMod")

    class device:
        r"""Context-manager that changes the selected device.

        Args:
            device (torch.device or int): device index to select. It's a no-op if
                this argument is a negative integer or ``None``.
        """

        def __init__(self, device):
            self.idx = torch.accelerator._get_device_index(device, optional=True)
            self.prev_idx = -1

        def __enter__(self):
            self.prev_idx = driver.exec("exchangeDevice", self.idx)

        def __exit__(self, type, value, traceback):
            self.idx = driver.exec("uncheckedSetDevice", self.prev_idx)
            return False

    def device_count() -> int:
        return driver.exec("deviceCount")

    def is_available():
        return True

    def current_device():
        return torch.accelerator.current_device_index()

    def get_rng_state(device):
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device("openreg", device)
        idx = device.index
        if idx is None:
            idx = current_device()
        default_generator = pytorch_openreg._C._get_default_generator(idx)
        return default_generator.get_state()

    def set_rng_state(new_state, device):
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device("openreg", device)
        idx = device.index
        if idx is None:
            idx = current_device()
        default_generator = pytorch_openreg._C._get_default_generator(idx)
        default_generator.set_state(new_state)

    def is_initialized():
        return module._initialized

    def _lazy_init():
        if is_initialized():
            return
        pytorch_openreg._C._init()
        module._initialized = True

    module.is_available = is_available  # type: ignore[assignment]

    module._initialized = False  # type: ignore[assignment]
    module._lazy_init = _lazy_init  # type: ignore[assignment]
    module.is_initialized = is_initialized  # type: ignore[assignment]

    module.device = device  # type: ignore[assignment]
    module.device_count = device_count  # type: ignore[assignment]
    module.current_device = current_device  # type: ignore[assignment]
    module.get_rng_state = get_rng_state  # type: ignore[assignment]
    module.set_rng_state = set_rng_state  # type: ignore[assignment]

    return module


# Set all the appropriate state on PyTorch
torch.utils.rename_privateuse1_backend("openreg")
torch._register_device_module("openreg", _create_module())
