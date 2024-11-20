# mypy: allow-untyped-defs
r"""
This package enables an interface for accessing MTIA backend in python
"""

import threading
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import device as _device, Tensor
from torch._utils import _dummy_type, _LazySeedTracker, classproperty
from torch.types import Device

from ._utils import _get_device_index


_device_t = Union[_device, str, int]

# torch.mtia.Event/Stream is alias of torch.Event/Stream
Event = torch.Event
Stream = torch.Stream

_initialized = False
_queued_calls: List[
    Tuple[Callable[[], None], List[str]]
] = []  # don't invoke these until initialization occurs
_tls = threading.local()
_initialization_lock = threading.Lock()
_lazy_seed_tracker = _LazySeedTracker()

rng_supported_mesh = True


def init():
    _lazy_init()


def is_initialized():
    r"""Return whether PyTorch's MTIA state has been initialized."""
    return _initialized and not _is_in_bad_fork()


def _is_in_bad_fork() -> bool:
    return torch._C._mtia_isInBadFork()


def _lazy_init() -> None:
    global _initialized, _queued_calls
    if is_initialized() or hasattr(_tls, "is_initializing"):
        return
    with _initialization_lock:
        # We be double-checking locking, boys! This is OK because
        # the above test was GIL protected anyway. The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if is_initialized():
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL
        if _is_in_bad_fork():
            raise RuntimeError(
                "Cannot re-initialize MTIA in forked subprocess. To use MTIA with "
                "multiprocessing, you must use the 'spawn' start method"
            )
        if not _is_compiled():
            raise AssertionError(
                "Torch not compiled with MTIA enabled. "
                "Ensure you have `import mtia.host_runtime.torch_mtia` in your python "
                "src file and include `//mtia/host_runtime/torch_mtia:torch_mtia` as "
                "your target dependency!"
            )

        torch._C._mtia_init()
        # Some of the queued calls may reentrantly call _lazy_init();
        # we need to just return without initializing in that case.
        # However, we must not let any *other* threads in!
        _tls.is_initializing = True

        _queued_calls.extend(calls for calls in _lazy_seed_tracker.get_calls() if calls)

        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = (
                        f"MTIA call failed lazily at initialization with error: {str(e)}\n\n"
                        f"MTIA call was originally invoked at:\n\n{''.join(orig_traceback)}"
                    )
                    raise DeferredMtiaCallError(msg) from e
        finally:
            delattr(_tls, "is_initializing")
        _initialized = True


class DeferredMtiaCallError(Exception):
    pass


def _is_compiled() -> bool:
    r"""Return true if compiled with MTIA support."""
    return torch._C._mtia_isBuilt()


def is_available() -> bool:
    r"""Return true if MTIA device is available"""
    if not _is_compiled():
        return False
    # MTIA has to init devices first to know if there is any devices available.
    return device_count() > 0


def synchronize(device: Optional[_device_t] = None) -> None:
    r"""Waits for all jobs in all streams on a MTIA device to complete."""
    with torch.mtia.device(device):
        return torch._C._mtia_deviceSynchronize()


def device_count() -> int:
    r"""Return the number of MTIA devices available."""
    return torch._C._accelerator_hooks_device_count()


def current_device() -> int:
    r"""Return the index of a currently selected device."""
    return torch._C._accelerator_hooks_get_current_device()


def current_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Return the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.mtia.current_device`, if :attr:`device` is ``None``
            (default).
    """
    return torch._C._mtia_getCurrentStream(_get_device_index(device, optional=True))


def default_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Return the default :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch.mtia.current_device`, if :attr:`device` is ``None``
            (default).
    """
    return torch._C._mtia_getDefaultStream(_get_device_index(device, optional=True))


def record_memory_history(
    enabled: Optional[str] = "all", stacks: str = "python", max_entries: int = 0
) -> None:
    r"""Enable/Disable the memory profiler on MTIA allocator

    Args:
        enabled (all or state, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).

        stacks ("python" or "cpp", optional). Select the stack trace to record.

        max_entries (int, optional). Maximum number of entries to record.
    """
    if not is_initialized():
        return
    torch._C._mtia_recordMemoryHistory(enabled, stacks, max_entries)


def snapshot() -> Dict[str, Any]:
    r"""Return a dictionary of MTIA memory allocator history"""

    return torch._C._mtia_memorySnapshot()


def get_device_capability(device: Optional[_device_t] = None) -> Tuple[int, int]:
    r"""Return capability of a given device as a tuple of (major version, minor version).

    Args:
        device (torch.device or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
    return torch._C._mtia_getDeviceCapability(_get_device_index(device, optional=True))


def empty_cache() -> None:
    r"""Empty the MTIA device cache."""
    return torch._C._mtia_emptyCache()


def set_stream(stream: Stream):
    r"""Set the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    torch._C._mtia_setCurrentStream(stream)


def set_device(device: _device_t) -> None:
    r"""Set the current device.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = _get_device_index(device)
    if device >= 0:
        torch._C._accelerator_hooks_set_current_device(device)


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch._C._accelerator_hooks_maybe_exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = torch._C._accelerator_hooks_maybe_exchange_device(self.prev_idx)
        return False


class StreamContext:
    r"""Context-manager that selects a given stream.

    All MTIA kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """

    cur_stream: Optional["torch.mtia.Stream"]

    def __init__(self, stream: Optional["torch.mtia.Stream"]):
        self.cur_stream = None
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                self.idx = -1

        self.src_prev_stream = (
            None if not torch.jit.is_scripting() else torch.mtia.default_stream(None)
        )
        self.dst_prev_stream = (
            None if not torch.jit.is_scripting() else torch.mtia.default_stream(None)
        )

    def __enter__(self):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None or MTIA device not available
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = torch.mtia.current_stream(None)

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = torch.mtia.current_stream(cur_stream.device)
        torch.mtia.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no MTIA device available, return
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            torch.mtia.set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        torch.mtia.set_stream(self.src_prev_stream)  # type: ignore[arg-type]


def stream(stream: Optional["torch.mtia.Stream"]) -> StreamContext:
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: In eager mode stream is of type Stream class while in JIT it doesn't support torch.mtia.stream
    """
    return StreamContext(stream)


def get_rng_state(device: Union[int, str, torch.device] = "mtia") -> Tensor:
    r"""Returns the random number generator state as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'mtia'`` (i.e., ``torch.device('mtia')``, the current mtia device).
    """
    warnings.warn(
        "get_rng_state is not implemented in torch.mtia",
        UserWarning,
        stacklevel=2,
    )
    return torch.zeros([1], dtype=torch.uint8, device=device)


def set_rng_state(
    new_state: Tensor, device: Union[int, str, torch.device] = "mtia"
) -> None:
    r"""Sets the random number generator state.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'mtia'`` (i.e., ``torch.device('mtia')``, the current mtia device).
    """
    warnings.warn(
        "set_rng_state is not implemented in torch.mtia",
        UserWarning,
        stacklevel=2,
    )


from .memory import *  # noqa: F403


__all__ = [
    "init",
    "is_available",
    "is_initialized",
    "synchronize",
    "device_count",
    "current_device",
    "current_stream",
    "default_stream",
    "memory_stats",
    "max_memory_allocated",
    "get_device_capability",
    "record_memory_history",
    "snapshot",
    "empty_cache",
    "set_device",
    "set_stream",
    "stream",
    "device",
    "set_rng_state",
    "get_rng_state",
]
