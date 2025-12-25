# mypy: allow-untyped-defs
r"""
This package enables an interface for accessing MTIA backend in python
"""

import threading
import traceback
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch._utils import _LazySeedTracker
from torch.types import Device

from ._utils import _get_device_index


# torch.mtia.Event/Stream is alias of torch.Event/Stream
Event = torch.Event
Stream = torch.Stream

# Default generators are initialized inside _mtia_init
default_generators: tuple[torch._C.Generator, ...] = ()  # type: ignore[assignment]


_initialized = False
_queued_calls: list[
    tuple[Callable[[], None], list[str]]
] = []  # don't invoke these until initialization occurs
_tls = threading.local()
_initialization_lock = threading.Lock()
_lazy_seed_tracker = _LazySeedTracker()


if hasattr(torch._C, "_mtia_exchangeDevice"):
    _exchange_device = torch._C._mtia_exchangeDevice
else:

    def _exchange_device(device: int) -> int:
        if device < 0:
            return -1
        raise RuntimeError("PyTorch was compiled without MTIA support")


if hasattr(torch._C, "_mtia_maybeExchangeDevice"):
    _maybe_exchange_device = torch._C._mtia_maybeExchangeDevice
else:

    def _maybe_exchange_device(device: int) -> int:
        if device < 0:
            return -1
        raise RuntimeError("PyTorch was compiled without MTIA support")


def init():
    _lazy_init()


def is_initialized():
    r"""Return whether PyTorch's MTIA state has been initialized."""
    return _initialized and not _is_in_bad_fork()


def _lazy_call(callable, **kwargs):
    with _initialization_lock:
        if is_initialized():
            return callable()
        else:
            global _lazy_seed_tracker
            if kwargs.get("seed_all", False):
                _lazy_seed_tracker.queue_seed_all(callable, traceback.format_stack())
            elif kwargs.get("seed", False):
                _lazy_seed_tracker.queue_seed(callable, traceback.format_stack())
            else:
                _queued_calls.append((callable, traceback.format_stack()))


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
                "Ensure you have `import mtia.host_runtime.torch_mtia.dynamic_library` in your python "
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


def synchronize(device: Device = None) -> None:
    r"""Waits for all jobs in all streams on a MTIA device to complete."""
    with torch.mtia.device(device):
        return torch._C._mtia_deviceSynchronize()


def device_count() -> int:
    r"""Return the number of MTIA devices available."""
    # TODO: Update _accelerator_hooks_device_count to abstract a MTIA device count API
    return torch._C._mtia_getDeviceCount()


def current_device() -> int:
    r"""Return the index of a currently selected device."""
    return torch._C._accelerator_hooks_get_current_device()


def current_stream(device: Device = None) -> Stream:
    r"""Return the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.mtia.current_device`, if :attr:`device` is ``None``
            (default).
    """
    return torch._C._mtia_getCurrentStream(_get_device_index(device, optional=True))


def default_stream(device: Device = None) -> Stream:
    r"""Return the default :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch.mtia.current_device`, if :attr:`device` is ``None``
            (default).
    """
    return torch._C._mtia_getDefaultStream(_get_device_index(device, optional=True))


def record_memory_history(
    enabled: str | None = "all", stacks: str = "python", max_entries: int = 0
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


def snapshot() -> dict[str, Any]:
    r"""Return a dictionary of MTIA memory allocator history"""

    return torch._C._mtia_memorySnapshot()


def attach_out_of_memory_observer(
    observer: Callable[[int, int, int, int], None],
) -> None:
    r"""Attach an out-of-memory observer to MTIA memory allocator"""
    torch._C._mtia_attachOutOfMemoryObserver(observer)


def is_bf16_supported(including_emulation: bool = True):
    r"""Return a bool indicating if the current MTIA device supports dtype bfloat16."""
    return True


def get_device_capability(device: Device = None) -> tuple[int, int]:
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
    r"""Set the current stream. This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    torch._C._mtia_setCurrentStream(stream)


def set_device(device: Device) -> None:
    r"""Set the current device.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = _get_device_index(device)
    if device >= 0:
        torch._C._accelerator_hooks_set_current_device(device)


def get_device_properties(device: Device = None) -> dict[str, Any]:
    r"""Return a dictionary of MTIA device properties

    Args:
        device (torch.device or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
    return torch._C._mtia_getDeviceProperties(_get_device_index(device, optional=True))


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

    cur_stream: Stream | None

    def __init__(self, stream: Stream | None):
        self.cur_stream = None
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                self.idx = -1  # pyrefly: ignore [bad-assignment]

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


def _set_stream_by_id(stream_id, device_index, device_type):
    r"""set stream specified by the stream id, device index and
        device type

    Args: stream_id (int): stream id in stream pool
          device_index (int): device index in topo
          device_type (int): enum device type
    """
    torch._C._mtia_setStream(stream_id, device_index, device_type)


def stream(stream: Stream | None) -> StreamContext:
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: In eager mode stream is of type Stream class while in JIT it doesn't support torch.mtia.stream
    """
    return StreamContext(stream)


def get_rng_state(device: Device = "mtia") -> Tensor:
    r"""Returns the random number generator state of the specified MTIA device as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'mtia'`` (i.e., ``torch.device('mtia')``, the current mtia device).

    .. warning::
        This function eagerly initializes MTIA.
    """
    _lazy_init()
    idx = _get_device_index(device, optional=True)
    if idx is None:
        idx = current_device()
    default_generator = default_generators[idx]
    return default_generator.get_state()


def get_rng_state_all() -> list[Tensor]:
    r"""Returns a list of ByteTensor representing the random number states of all devices."""
    results = [get_rng_state(i) for i in range(device_count())]
    return results


def set_rng_state(new_state: Tensor, device: Device = "mtia") -> None:
    r"""Sets the random number generator state of the specified MTIA device.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'mtia'`` (i.e., ``torch.device('mtia')``, the current mtia device).
    """
    if not is_initialized():
        with torch._C._DisableFuncTorch():
            # Clone the state because the callback will be triggered
            # later when MTIA is lazy initialized.
            new_state = new_state.clone(memory_format=torch.contiguous_format)

    idx = _get_device_index(device, optional=True)
    if idx is None:
        idx = current_device()

    def cb():
        default_generator = default_generators[idx]
        default_generator.set_state(new_state)

    _lazy_call(cb)


def set_rng_state_all(new_states: list[Tensor]) -> None:
    r"""Sets the random number generator state of all devices.

    Args:
        new_states (Iterable of torch.ByteTensor): The desired state for each device.
    """
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


def manual_seed(seed: int) -> None:
    r"""Sets the seed for generating random numbers for the current MTIA device.
    It's safe to call this function if MTIA is not available; in that case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    """
    seed = int(seed)

    def cb():
        idx = current_device()
        default_generator = default_generators[idx]
        default_generator.manual_seed(seed)

    _lazy_call(cb, seed=True)


def manual_seed_all(seed: int) -> None:
    r"""Sets the seed for generating random numbers on all MTIA devices.
    It's safe to call this function if MTIA is not available; in that case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)

    def cb():
        for i in range(device_count()):
            default_generator = default_generators[i]
            default_generator.manual_seed(seed)

    _lazy_call(cb, seed_all=True)


def seed() -> int:
    r"""Sets the seed for generating random numbers to a random number for the current MTIA device.
    It's safe to call this function if MTIA is not available; in that case, it is silently ignored.

    .. warning::
        If you are working with a multi-GPU model, this function will only initialize
        the seed on one GPU.  To initialize all GPUs, use :func:`seed_all`.
    """

    def cb():
        idx = current_device()
        default_generator = default_generators[idx]
        return default_generator.seed()

    return _lazy_call(cb, seed=True)


def seed_all() -> int:
    r"""Sets the seed for generating random numbers to a random number on all MTIA devices.

    It's safe to call this function if MTIA is not available; in that case, it is silently ignored.
    """

    def cb():
        random_seed = 0
        seeded = False
        for i in range(device_count()):
            default_generator = default_generators[i]
            if not seeded:
                random_seed = default_generator.seed()
                seeded = True
            else:
                default_generator.manual_seed(random_seed)
        return random_seed

    return _lazy_call(cb, seed_all=True)


def initial_seed() -> int:
    r"""Returns the current random seed of the current MTIA device.

    .. warning::
        This function eagerly initializes MTIA.
    """
    _lazy_init()
    idx = current_device()
    return default_generators[idx].initial_seed()


from .memory import *  # noqa: F403
from .mtia_graph import *  # noqa: F403


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
    "memory_allocated",
    "reset_peak_memory_stats",
    "get_device_capability",
    "get_device_properties",
    "record_memory_history",
    "snapshot",
    "attach_out_of_memory_observer",
    "empty_cache",
    "set_device",
    "set_stream",
    "stream",
    "device",
    "set_rng_state",
    "get_rng_state",
    "set_rng_state_all",
    "get_rng_state_all",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
    "is_bf16_supported",
    "MTIAGraph",
    "graph",
    "graph_pool_handle",
]
