# mypy: allow-untyped-defs
r"""
This package introduces support for the XPU backend, specifically tailored for
Intel GPU optimization.

This package is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports XPU.
"""

import threading
import traceback
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Optional

import torch
import torch._C
from torch._utils import _dummy_type, _LazySeedTracker
from torch.types import Device

from ._utils import _get_device_index
from .streams import Event, Stream


_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls: list[
    tuple[Callable[[], None], list[str]]
] = []  # don't invoke these until initialization occurs
_is_in_bad_fork = getattr(torch._C, "_xpu_isInBadFork", lambda: False)
_lazy_seed_tracker = _LazySeedTracker()
default_generators: tuple[torch._C.Generator] = ()  # type: ignore[assignment]


def _is_compiled() -> bool:
    r"""Return true if compile with XPU support."""
    return torch._C._has_xpu


if _is_compiled():
    _XpuDeviceProperties = torch._C._XpuDeviceProperties
    _exchange_device = torch._C._xpu_exchangeDevice
    _maybe_exchange_device = torch._C._xpu_maybeExchangeDevice
else:
    # Define dummy if PyTorch was compiled without XPU
    _XpuDeviceProperties = _dummy_type("_XpuDeviceProperties")  # type: ignore[assignment, misc]

    def _exchange_device(device: int) -> int:
        raise NotImplementedError("PyTorch was compiled without XPU support")

    def _maybe_exchange_device(device: int) -> int:
        raise NotImplementedError("PyTorch was compiled without XPU support")


@lru_cache(maxsize=1)
def device_count() -> int:
    r"""Return the number of XPU device available."""
    if not _is_compiled():
        return 0
    return torch._C._xpu_getDeviceCount()


def is_available() -> bool:
    r"""Return a bool indicating if XPU is currently available."""
    # This function never throws.
    return device_count() > 0


def is_bf16_supported(including_emulation: bool = True) -> bool:
    r"""Return a bool indicating if the current XPU device supports dtype bfloat16."""
    if not is_available():
        return False
    return (
        including_emulation
        or torch.xpu.get_device_properties().has_bfloat16_conversions
    )


def is_tf32_supported() -> bool:
    r"""Return a bool indicating if the current XPU device supports dtype tf32."""
    if not is_available():
        return False
    # On Intel Xe architecture and newer, TF32 operations can be accelerated
    # through DPAS (Dot Product Accumulate Systolic) instructions. Therefore,
    # TF32 support can be determined by checking whether the device supports
    # subgroup matrix multiply-accumulate operations.
    return torch.xpu.get_device_properties().has_subgroup_matrix_multiply_accumulate


def is_initialized():
    r"""Return whether PyTorch's XPU state has been initialized."""
    return _initialized and not _is_in_bad_fork()


def _lazy_call(callable, **kwargs) -> None:
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


def init() -> None:
    r"""Initialize PyTorch's XPU state.
    This is a Python API about lazy initialization that avoids initializing
    XPU until the first time it is accessed. Does nothing if the XPU state is
    already initialized.
    """
    _lazy_init()


def _lazy_init() -> None:
    global _initialized, _queued_calls
    if is_initialized() or hasattr(_tls, "is_initializing"):
        return
    with _initialization_lock:
        # This test was was protected via GIL. Double-check whether XPU has
        # already been initialized.
        if is_initialized():
            return
        # Stop promptly upon encountering a bad fork error.
        if _is_in_bad_fork():
            raise RuntimeError(
                "Cannot re-initialize XPU in forked subprocess. To use XPU with "
                "multiprocessing, you must use the 'spawn' start method"
            )
        if not _is_compiled():
            raise AssertionError("Torch not compiled with XPU enabled")
        # This function inits XPU backend and detects bad fork processing.
        torch._C._xpu_init()
        # Some of the queued calls may reentrantly call _lazy_init(); We need to
        # just return without initializing in that case.
        _tls.is_initializing = True

        _queued_calls.extend(calls for calls in _lazy_seed_tracker.get_calls() if calls)

        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = (
                        f"XPU call failed lazily at initialization with error: {str(e)}\n\n"
                        f"XPU call was originally invoked at:\n\n{''.join(orig_traceback)}"
                    )
                    raise Exception(msg) from e  # noqa: TRY002
        finally:
            delattr(_tls, "is_initializing")
        _initialized = True


class _DeviceGuard:
    def __init__(self, index: int) -> None:
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch.xpu._exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = torch.xpu._maybe_exchange_device(self.prev_idx)
        return False


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int or str): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any) -> None:
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch.xpu._exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = torch.xpu._maybe_exchange_device(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a XPU, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj) -> None:
        idx = obj.get_device() if obj.is_xpu else -1
        super().__init__(idx)


def set_device(device: Device) -> None:
    r"""Set the current device.

    Args:
        device (torch.device or int or str): selected device. This function is a
            no-op if this argument is negative.
    """
    _lazy_init()
    device = _get_device_index(device)
    if device >= 0:
        torch._C._xpu_setDevice(device)


def get_device_name(device: Device = None) -> str:
    r"""Get the name of a device.

    Args:
        device (torch.device or int or str, optional): device for which to
            return the name. This function is a no-op if this argument is a
            negative integer. It uses the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """
    return get_device_properties(device).name


@lru_cache(None)
def get_device_capability(device: Device = None) -> dict[str, Any]:
    r"""Get the xpu capability of a device.

    Args:
        device (torch.device or int or str, optional): device for which to
            return the device capability. This function is a no-op if this
            argument is a negative integer. It uses the current device, given by
            :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        dict[str, Any]: the xpu capability dictionary of the device
    """
    props = get_device_properties(device)
    # Only keep attributes that are safe for dictionary serialization.
    serializable_types = (int, float, bool, str, type(None), list, tuple, dict)
    return {
        key: value
        for key in dir(props)
        if not key.startswith("__")
        and isinstance((value := getattr(props, key)), serializable_types)
    }


def get_device_properties(
    device: Device = None,
) -> _XpuDeviceProperties:  # pyrefly: ignore  # not-a-type
    r"""Get the properties of a device.

    Args:
        device (torch.device or int or str): device for which to return the
            properties of the device.

    Returns:
        _XpuDeviceProperties: the properties of the device
    """
    _lazy_init()
    device = _get_device_index(device, optional=True)
    return _get_device_properties(device)  # type: ignore[name-defined]  # noqa: F821


def current_device() -> int:
    r"""Return the index of a currently selected device."""
    _lazy_init()
    return torch._C._xpu_getDevice()


def _get_device(device: int | str | torch.device) -> torch.device:
    r"""Return the torch.device type object from the passed in device.

    Args:
        device (torch.device or int or str): selected device.
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("xpu", device)
    return device


def can_device_access_peer(device: Device, peer: Device) -> bool:
    r"""Query whether a device can access a peer device's memory.

    Args:
        device (torch.device or int or str): selected device.
        peer (torch.device or int or str): peer device to query access to.

    Returns:
        bool: ``True`` if ``device`` can access ``peer``, ``False`` otherwise.
    """
    _lazy_init()
    device = _get_device_index(device, optional=True)
    peer = _get_device_index(peer, optional=True)
    return torch._C._xpu_canDeviceAccessPeer(device, peer)


class StreamContext:
    r"""Context-manager that selects a given stream.

    All XPU kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """

    cur_stream: Optional["torch.xpu.Stream"]

    def __init__(self, stream: Optional["torch.xpu.Stream"]) -> None:
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if self.idx is None:
            self.idx = -1  # pyrefly: ignore [bad-assignment]

    def __enter__(self):
        cur_stream = self.stream
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = torch.xpu.current_stream(None)

        # If the stream is not on the current device, then set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = torch.xpu.current_stream(cur_stream.device)
        torch.xpu.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        cur_stream = self.stream
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device and destination device
        if self.src_prev_stream.device != cur_stream.device:
            torch.xpu.set_stream(self.dst_prev_stream)
        torch.xpu.set_stream(self.src_prev_stream)


def stream(stream: Optional["torch.xpu.Stream"]) -> StreamContext:
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's ``None``.
    """
    return StreamContext(stream)


def _set_stream_by_id(stream_id, device_index, device_type) -> None:
    r"""set stream specified by the stream id, device index and device type

    Args: stream_id (int): not visible to the user, used to assigned to the specific stream.
          device_index (int): selected device index.
          device_type (int): selected device type.
    """
    torch._C._xpu_setStream(
        stream_id=stream_id,
        device_index=device_index,
        device_type=device_type,
    )


def set_stream(stream: Stream) -> None:
    r"""Set the current stream. This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    _lazy_init()
    _set_stream_by_id(
        stream_id=stream.stream_id,
        device_index=stream.device_index,
        device_type=stream.device_type,
    )


def current_stream(device: Device = None) -> Stream:
    r"""Return the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.xpu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    streamdata = torch._C._xpu_getCurrentStream(
        _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def get_stream_from_external(data_ptr: int, device: Device = None) -> Stream:
    r"""Return a :class:`Stream` from an external SYCL queue.

    This function is used to wrap SYCL queue created in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This function doesn't manage the queue life-cycle, it is the user
       responsibility to keep the referenced queue alive while this returned stream is
       being used. The different SYCL queue pointers will result in distinct
       :class:`Stream` objects, even if the SYCL queues they dereference are equivalent.

    Args:
        data_ptr(int): Integer representation of the `sycl::queue*` value passed externally.
        device(torch.device or int, optional): the device where the queue was originally created.
            It is the user responsibility to ensure the device is specified correctly.
    """
    _lazy_init()
    streamdata = torch._C._xpu_getStreamFromExternal(
        data_ptr, _get_device_index(device, optional=True)
    )
    return Stream(
        stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2]
    )


def synchronize(device: Device = None) -> None:
    r"""Wait for all kernels in all streams on a XPU device to complete.

    Args:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.xpu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    _lazy_init()
    device = _get_device_index(device, optional=True)
    return torch._C._xpu_synchronize(device)


def get_arch_list() -> list[str]:
    r"""Return list XPU architectures this library was compiled for."""
    if not _is_compiled():
        return []
    arch_flags = torch._C._xpu_getArchFlags()
    if arch_flags is None:
        return []
    return arch_flags.split()


def get_gencode_flags() -> str:
    r"""Return XPU AOT(ahead-of-time) build flags this library was compiled with."""
    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return ""
    return f"-device {','.join(arch for arch in arch_list)}"


def _get_generator(device: torch.device) -> torch._C.Generator:
    r"""Return the XPU Generator object for the given device.

    Args:
        device (torch.device): selected device.
    """
    idx = device.index
    if idx is None:
        idx = current_device()
    return torch.xpu.default_generators[idx]


def _set_rng_state_offset(
    offset: int, device: int | str | torch.device = "xpu"
) -> None:
    r"""Set the random number generator state offset of the specified GPU.

    Args:
        offset (int): The desired offset
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).
    """
    final_device = _get_device(device)

    def cb() -> None:
        default_generator = _get_generator(final_device)
        default_generator.set_offset(offset)

    _lazy_call(cb)


def _get_rng_state_offset(device: int | str | torch.device = "xpu") -> int:
    r"""Return the random number generator state offset of the specified GPU.

    Args:
        device (torch.device or int, optional): The device to return the RNG state offset of.
            Default: ``'xpu'`` (i.e., ``torch.device('xpu')``, the current XPU device).

    .. warning::
        This function eagerly initializes XPU.
    """
    _lazy_init()
    final_device = _get_device(device)
    default_generator = _get_generator(final_device)
    return default_generator.get_offset()


# import here to avoid circular import
from .memory import (
    change_current_allocator,
    empty_cache,
    get_per_process_memory_fraction,
    max_memory_allocated,
    max_memory_reserved,
    mem_get_info,
    memory_allocated,
    memory_reserved,
    memory_snapshot,
    memory_stats,
    memory_stats_as_nested_dict,
    MemPool,
    reset_accumulated_memory_stats,
    reset_peak_memory_stats,
    set_per_process_memory_fraction,
    use_mem_pool,
    XPUPluggableAllocator,
)
from .random import (
    get_rng_state,
    get_rng_state_all,
    initial_seed,
    manual_seed,
    manual_seed_all,
    seed,
    seed_all,
    set_rng_state,
    set_rng_state_all,
)


__all__ = [
    "Event",
    "Stream",
    "StreamContext",
    "XPUPluggableAllocator",
    "can_device_access_peer",
    "change_current_allocator",
    "current_device",
    "current_stream",
    "default_generators",
    "device",
    "device_of",
    "device_count",
    "empty_cache",
    "get_arch_list",
    "get_device_capability",
    "get_device_name",
    "get_device_properties",
    "get_gencode_flags",
    "get_per_process_memory_fraction",
    "get_rng_state",
    "get_rng_state_all",
    "get_stream_from_external",
    "init",
    "initial_seed",
    "is_available",
    "is_bf16_supported",
    "is_initialized",
    "is_tf32_supported",
    "manual_seed",
    "manual_seed_all",
    "max_memory_allocated",
    "max_memory_reserved",
    "mem_get_info",
    "memory_allocated",
    "memory_reserved",
    "memory_snapshot",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "MemPool",
    "use_mem_pool",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "seed",
    "seed_all",
    "set_device",
    "set_per_process_memory_fraction",
    "set_rng_state",
    "set_rng_state_all",
    "set_stream",
    "stream",
    "streams",
    "synchronize",
]
