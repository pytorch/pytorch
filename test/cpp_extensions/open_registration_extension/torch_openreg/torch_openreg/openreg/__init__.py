import torch

import torch_openreg._C  # type: ignore[misc]
from . import meta
from .amp import get_amp_supported_dtype


_initialized = False
_is_in_bad_fork = getattr(torch_openreg._C, "_isInBadFork", lambda: False)


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
        self.prev_idx = torch_openreg._C._exchangeDevice(self.idx)

    def __exit__(self, type, value, traceback):
        self.idx = torch_openreg._C._set_device(self.prev_idx)
        return False


def is_available():
    return True


def device_count() -> int:
    return torch_openreg._C._get_device_count()


def current_device():
    return torch_openreg._C._get_device()


# LITERALINCLUDE START: PYTHON SET DEVICE FUNCTION
def set_device(device) -> None:
    if device >= 0:
        torch_openreg._C._set_device(device)


# LITERALINCLUDE END: PYTHON SET DEVICE FUNCTION


def current_stream(device=None):
    """Return the current stream for the device.

    Under torch.compile this is traced from Dynamo's symbolic stream state;
    this implementation is used in eager mode.
    """
    return torch.accelerator.current_stream(device)


def synchronize(device=None):
    """Wait for all kernels on the device to complete.

    Under torch.compile this is traced as torch.ops.streams.synchronize_device;
    this implementation is used in eager mode.
    """
    return torch.accelerator.synchronize(device)


def _exchange_device(device_index):
    return torch_openreg._C._exchangeDevice(device_index)


def _maybe_exchange_device(device_index):
    # Negative indices report no exchange via the -1 convention (matching
    # torch.cuda); positive out-of-range indices raise, like the C++ backend's
    # maybe_exchange_device and set_device do.
    if device_index < 0:
        return -1
    count = device_count()
    if device_index >= count:
        raise RuntimeError(
            "The device index is out of range. It must be in [0, "
            f"{count}), but got {device_index}."
        )
    return torch_openreg._C._exchangeDevice(device_index)


def init():
    _lazy_init()


def is_initialized():
    return _initialized and not _is_in_bad_fork()


def _lazy_init():
    global _initialized
    if is_initialized():
        return
    if _is_in_bad_fork():
        raise RuntimeError(
            "Cannot re-initialize OpenReg in forked subprocess. To use OpenReg with "
            "multiprocessing, you must use the 'spawn' start method"
        )
    torch_openreg._C._init()
    _initialized = True


from .random import *  # noqa: F403


__all__ = [
    "device",
    "device_count",
    "current_device",
    "current_stream",
    "set_device",
    "synchronize",
    "initial_seed",
    "is_available",
    "init",
    "is_initialized",
    "random",
    "manual_seed",
    "manual_seed_all",
    "get_rng_state",
    "set_rng_state",
]
