import traceback
from collections.abc import Callable

import torch
from torch._utils import _LazySeedTracker

import torch_openreg._C  # type: ignore[misc]
from . import meta  # noqa: F401
from .amp import get_amp_supported_dtype  # noqa: F401


_initialized = False
_is_in_bad_fork = getattr(torch_openreg._C, "_isInBadFork", lambda: False)
_queued_calls: list[
    tuple[Callable[[], None], list[str]]
] = []  # don't invoke these until initialization occurs
_lazy_seed_tracker = _LazySeedTracker()


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


def init():
    _lazy_init()


def is_initialized():
    return _initialized and not _is_in_bad_fork()


def _lazy_call(callable, **kwargs):
    if is_initialized():
        callable()
    else:
        global _lazy_seed_tracker
        if kwargs.get("seed_all", False):
            _lazy_seed_tracker.queue_seed_all(callable, traceback.format_stack())
        elif kwargs.get("seed", False):
            _lazy_seed_tracker.queue_seed(callable, traceback.format_stack())
        else:
            # Don't store the actual traceback to avoid memory cycle
            _queued_calls.append((callable, traceback.format_stack()))


def _lazy_init():
    global _initialized, _queued_calls
    if is_initialized():
        return
    if _is_in_bad_fork():
        raise RuntimeError(
            "Cannot re-initialize OpenReg in forked subprocess. To use OpenReg with "
            "multiprocessing, you must use the 'spawn' start method"
        )
    torch_openreg._C._init()

    _queued_calls.extend(call for call in _lazy_seed_tracker.get_calls() if call)

    for queued_call, origin_trackback in _queued_calls:
        try:
            queued_call()
        except Exception as e:
            msg = (
                "Error during lazy initialization call from:\n"
                + "".join(origin_trackback)
                + f"\nError message: {e}"
            )
            raise Exception(msg) from e  # noqa: TRY002

    _initialized = True


from .random import *  # noqa: F403


__all__ = [
    "device",
    "device_count",
    "current_device",
    "set_device",
    "initial_seed",
    "is_available",
    "init",
    "is_initialized",
    "random",
    "manual_seed",
    "manual_seed_all",
    "get_rng_state",
    "get_rng_state_all",
    "seed",
    "seed_all",
    "set_rng_state",
    "set_rng_state_all",
]
