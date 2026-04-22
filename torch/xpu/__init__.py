# mypy: allow-untyped-defs
r"""
This package introduces support for the XPU backend, specifically tailored for
Intel GPU optimization.

This package is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports XPU.
"""

from __future__ import annotations

import os
import threading
import traceback
import warnings
from functools import lru_cache
from typing import Any, NewType, TYPE_CHECKING

import torch
import torch._C
from torch._utils import _dummy_type, _LazySeedTracker


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.types import Device

from ._utils import _get_device_index
from .graphs import (
    graph,
    graph_pool_handle,
    is_current_stream_capturing,
    make_graphed_callables,
    XPUGraph,
)
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
_cached_device_count: int | None = None


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


def _parse_visible_devices() -> list[int]:
    r"""Parse ``ZE_AFFINITY_MASK`` and return visible device ordinals.

    Returns a list of non-negative device ordinals specified by the mask.
    When the mask is unset, returns ``[0, 1, ..., 127]`` (the maximum range
    for ``int8_t`` device indices).  Returns an empty list for unsupported
    COMPOSITE-style masks (e.g. ``"0.0,0.1"``).
    """
    var = os.getenv("ZE_AFFINITY_MASK")
    if var is None:
        # DeviceIndex is stored as int8_t, so valid indices are 0–127
        # (up to 128 devices). Return the full range when no mask is set.
        return list(range(128))

    visible_devices: list[int] = []
    for elem in var.split(","):
        try:
            x = int(elem.strip())
        except ValueError:
            # A non-integer token (e.g. "0.0" in COMPOSITE-mode format)
            # means the mask is unsupported here; signal that by returning
            # an empty list.
            return []
        if x >= 0 and x not in visible_devices:
            visible_devices.append(x)
    return visible_devices


def _raw_device_count_zes(visible_mask: list[int]) -> int:
    r"""Return the number of visible XPU devices via Level Zero Sysman.

    Enumerates devices from the first Level Zero Sysman driver and counts those
    whose logical index appears in *visible_mask*.  Only devices listed in
    the visible mask participate in counting.

    Discrete GPUs (dGPUs) take priority: if any visible dGPU is found, only
    dGPUs are counted; integrated GPUs (iGPUs) are counted only when no
    visible dGPU exists.

    For tiled dGPUs (``numSubdevices > 0``), the counting depends on
    ``ZE_FLAT_DEVICE_HIERARCHY``:

    - **FLAT / COMBINED** (default): each sub-device is exposed as a
      separate top-level device and counted individually.
    - **COMPOSITE**: sub-devices are hidden; the whole physical device
      counts as one.

    Returns a negative value on initialization or enumeration failure.
    """
    from ctypes import byref, c_uint32

    try:
        import pyzes  # type: ignore[import]
    except ImportError:
        return -1

    def _zes_check(rc: int, msg: str) -> bool:
        """Return True if the call failed (rc != 0) after issuing a warning."""
        if rc != 0:
            warnings.warn(msg, stacklevel=3)
        return rc != 0

    if _zes_check(pyzes.zesInit(0), "Can't initialize Level Zero Sysman"):
        return -1

    driver_count = c_uint32(0)
    if _zes_check(
        pyzes.zesDriverGet(byref(driver_count), None),
        "Can't get Level Zero Sysman driver count",
    ):
        return -1
    if driver_count.value == 0:
        return 0

    drivers = (pyzes.zes_driver_handle_t * driver_count.value)()
    if _zes_check(
        pyzes.zesDriverGet(byref(driver_count), drivers),
        "Can't get Level Zero Sysman driver handles",
    ):
        return -1

    device_count = c_uint32(0)
    if _zes_check(
        pyzes.zesDeviceGet(drivers[0], byref(device_count), None),
        "Can't get Level Zero Sysman device count",
    ):
        return -1

    devices = (pyzes.zes_device_handle_t * device_count.value)()
    if _zes_check(
        pyzes.zesDeviceGet(drivers[0], byref(device_count), devices),
        "Can't get Level Zero Sysman device handles",
    ):
        return -1

    # --- Count visible dGPUs and iGPUs ---
    ZE_DEVICE_PROPERTY_FLAG_INTEGRATED = 1 << 0
    hierarchy = os.getenv("ZE_FLAT_DEVICE_HIERARCHY")
    expose_sub_devices = hierarchy != "COMPOSITE"

    visible = set(visible_mask)
    logical_index = 0
    num_igpu = 0
    num_dgpu = 0

    for device in devices:
        props = pyzes.zes_device_properties_t()
        props.stype = pyzes.ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES
        if _zes_check(
            pyzes.zesDeviceGetProperties(device, byref(props)),
            "Can't get Level Zero Sysman device properties",
        ):
            return -1

        is_integrated = bool(props.core.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED)

        # Determine how many logical indices this physical device occupies.
        # Tiled dGPUs in FLAT/COMBINED mode expose each sub-device separately;
        # everything else (iGPU, non-tiled dGPU, COMPOSITE mode) counts as one.
        num_slots = (
            props.numSubdevices
            if not is_integrated and props.numSubdevices > 0 and expose_sub_devices
            else 1
        )

        for _ in range(num_slots):
            if logical_index in visible:
                if is_integrated:
                    num_igpu += 1
                else:
                    num_dgpu += 1
            logical_index += 1

    # Prefer dGPU count; fall back to iGPU count only when no dGPU is visible.
    return num_dgpu or num_igpu


def _device_count_zes() -> int:
    r"""Return the number of visible XPU devices, or -1 on failure."""
    visible_devices = _parse_visible_devices()
    if not visible_devices:
        return -1
    return _raw_device_count_zes(visible_devices)


def device_count() -> int:
    r"""
    Return the number of XPU device available.

    .. note:: This API will NOT poison fork if Level Zero Sysman discovery succeeds.
        See :ref:`multiprocessing-poison-fork-note` for more details.
    """
    if not _is_compiled():
        return 0
    global _cached_device_count
    if _cached_device_count is not None:
        return _cached_device_count
    if _initialized or hasattr(_tls, "is_initializing"):
        count = torch._C._xpu_getDeviceCount()
    else:
        zes_count = _device_count_zes()
        count = torch._C._xpu_getDeviceCount() if zes_count < 0 else zes_count
    # Do not cache the device count prior to XPU initialization, because
    # the number of devices can change due to changes to ZE_AFFINITY_MASK
    # setting prior to XPU initialization.
    if _initialized:
        _cached_device_count = count
    return count


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
) -> _XpuDeviceProperties:
    r"""Get the properties of a device. Returns _XpuDeviceProperties containing the following device properties:

    - ``name`` (str): device name.
    - ``platform_name`` (str): SYCL platform name.
    - ``vendor`` (str): device vendor.
    - ``device_id`` (int): device identifier (product ID).
    - ``driver_version`` (str): driver version.
    - ``version`` (str): runtime version.
    - ``max_compute_units`` (int): number of parallel compute units.
    - ``gpu_eu_count`` (int): number of EUs (Execution Unit).
    - ``max_work_group_size``: (int): maximum number of work-items permitted in a work-group.
    - ``max_num_sub_groups`` (int): maximum number of sub-groups supported in a work-group.
    - ``memory_clock_rate`` (int) maximum clock rate of device's global memory in MHz.
    - ``memory_bus_width`` (int) maximum bus width between device and memory in bits.
    - ``sub_group_sizes``: (list[int]): a list of supported sub-group sizes.
    - ``local_mem_size`` (int): device local memory capacity that can be allocated per work-group in bytes.
    - ``has_fp16`` (bool): whether float16 dtype is supported.
    - ``has_fp64`` (bool): whether float64 dtype is supported.
    - ``has_atomic64`` (bool): whether 64-bit atomic operations are supported.
    - ``has_bfloat16_conversions`` (bool): whether bfloat16 conversions are supported.
    - ``has_subgroup_matrix_multiply_accumulate`` (bool): whether DPAS (Dot Product Accumulate Systolic) is supported.
    - ``has_subgroup_matrix_multiply_accumulate_tensor_float32`` (bool): whether DPAS with tf32 inputs is supported.
    - ``has_subgroup_2d_block_io`` (bool): whether 2D block I/O for efficient matrix multiplication is supported.
    - ``total_memory`` (int): device global memory in bytes.
    - ``gpu_subslice_count`` (int): number of subslice.
    - ``architecture`` (int): device architecture identifier (experimental).
    - ``type`` (str): device type, e.g. 'cpu', 'gpu', accelerator', 'host', 'unknown'.
    - ``uuid`` (Any): device UUID (Universal Unique ID), 16 bytes.

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

    cur_stream: torch.xpu.Stream | None

    def __init__(self, stream: torch.xpu.Stream | None) -> None:
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


def stream(stream: torch.xpu.Stream | None) -> StreamContext:
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


_POOL_HANDLE = NewType("_POOL_HANDLE", tuple[int, int])
__all__ = [
    "Event",
    "Stream",
    "StreamContext",
    "XPUPluggableAllocator",
    "XPUGraph",
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
    "graph",
    "graph_pool_handle",
    "init",
    "initial_seed",
    "is_available",
    "is_bf16_supported",
    "is_current_stream_capturing",
    "is_initialized",
    "is_tf32_supported",
    "make_graphed_callables",
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
