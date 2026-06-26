# mypy: allow-untyped-defs
r"""
This package introduces support for the XPU backend, specifically tailored for
Intel GPU optimization.

This package is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports XPU.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import traceback
import warnings
from ctypes import byref, c_double, c_uint32, c_void_p, cast, pointer
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


@dataclasses.dataclass
class _ZesDeviceInfo:
    device_handle: c_void_p
    subdevice_id: int | None = None
    is_integrated: bool = False
    temperature_handle: c_void_p | None = None
    frequency_handle: c_void_p | None = None
    power_handle: c_void_p | None = None
    engine_handle: c_void_p | None = None
    memory_handle: c_void_p | None = None


_cached_zes_device_infos: list[_ZesDeviceInfo] = []
# Interval between two HW counter reads; must be >=100ms for fresh data.
_zes_sample_interval_ms = 150


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


def _parse_visible_devices(strict=False) -> list[int]:
    r"""Parse ``ZE_AFFINITY_MASK`` and return visible device ordinals.

    Returns a list of non-negative device ordinals specified by the mask.
    When the mask is unset, returns ``[0, 1, ..., 127]`` (the maximum range
    for ``int8_t`` device indices).  Returns an empty list for unsupported
    COMPOSITE-style masks (e.g. ``"0.0,0.1"``).

    Args:
        strict (bool): If ``True``, raises ``ValueError`` on unsupported mask
            formats (e.g. COMPOSITE-style ``"0.0,0.1"``).  If ``False``
            (default), returns an empty list instead.
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
            if strict:
                raise ValueError(
                    f"Unsupported ZE_AFFINITY_MASK format: '{var}'. Expected a comma-separated list of integers, e.g. '0,1,2'."
                ) from None
            return []
        if x >= 0 and x not in visible_devices:
            visible_devices.append(x)
    return visible_devices


def _enum_zes_device_infos(visible_mask: list[int]) -> int:
    r"""Enumerate visible XPU devices via Level Zero Sysman and cache their info.

    Enumerates devices from the first Level Zero Sysman driver and counts those
    whose logical index appears in *visible_mask*.  Only devices listed in
    the visible mask participate in counting.
    The populated ``_cached_zes_device_infos`` list is indexed by PyTorch
    device ordinal.

    Discrete GPUs (dGPUs) take priority: if any visible dGPU is found, only
    dGPUs are counted; integrated GPUs (iGPUs) are counted only when no
    visible dGPU exists.

    For tiled dGPUs (``numSubdevices > 0``), the counting depends on
    ``ZE_FLAT_DEVICE_HIERARCHY``:

    - **FLAT / COMBINED** (default): each sub-device is exposed as a
      separate top-level device and counted individually.
    - **COMPOSITE**: sub-devices are hidden; the whole physical device
      counts as one.

    Returns the visible device count, or a negative value on failure.
    """
    try:
        import pyzes  # type: ignore[import]
    except Exception:
        return -1

    global _cached_zes_device_infos

    def _zes_check_warn(rc: int, msg: str) -> bool:
        """Return True if the call failed (rc != ZE_RESULT_SUCCESS) after issuing a warning."""
        if rc != pyzes.ZE_RESULT_SUCCESS:
            warnings.warn(msg, stacklevel=3)
        return rc != pyzes.ZE_RESULT_SUCCESS

    if _zes_check_warn(pyzes.zesInit(0), "Can't initialize Level Zero Sysman"):
        return -1

    driver_count = c_uint32(0)
    if _zes_check_warn(
        pyzes.zesDriverGet(byref(driver_count), None),
        "Can't get Level Zero Sysman driver count",
    ):
        return -1
    if driver_count.value == 0:
        return 0

    drivers = (pyzes.zes_driver_handle_t * driver_count.value)()
    if _zes_check_warn(
        pyzes.zesDriverGet(byref(driver_count), drivers),
        "Can't get Level Zero Sysman driver handles",
    ):
        return -1

    device_count = c_uint32(0)
    if _zes_check_warn(
        pyzes.zesDeviceGet(drivers[0], byref(device_count), None),
        "Can't get Level Zero Sysman device count",
    ):
        return -1

    devices = (pyzes.zes_device_handle_t * device_count.value)()
    if _zes_check_warn(
        pyzes.zesDeviceGet(drivers[0], byref(device_count), devices),
        "Can't get Level Zero Sysman device handles",
    ):
        return -1

    # --- Count visible dGPUs and iGPUs ---
    ZES_DEVICE_PROPERTY_FLAG_INTEGRATED = 1 << 0
    expose_subdevices = os.getenv("ZE_FLAT_DEVICE_HIERARCHY") != "COMPOSITE"

    _cached_zes_device_infos.clear()
    visible = set(visible_mask)
    logical_index = 0
    num_igpu = 0
    num_dgpu = 0

    for device in devices:
        props = pyzes.zes_device_properties_t()
        props.stype = pyzes.ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES
        ext_props = pyzes.zes_device_ext_properties_t()
        ext_props.stype = pyzes.ZES_STRUCTURE_TYPE_DEVICE_EXT_PROPERTIES
        props.pNext = cast(pointer(ext_props), c_void_p)
        if _zes_check_warn(
            pyzes.zesDeviceGetProperties(device, byref(props)),
            "Can't get Level Zero Sysman device properties",
        ):
            return -1

        is_integrated = bool(ext_props.flags & ZES_DEVICE_PROPERTY_FLAG_INTEGRATED)

        # Tiled dGPUs in FLAT/COMBINED mode expose each sub-device as a
        # separate logical device; everything else counts as one slot.
        tiled = not is_integrated and props.numSubdevices > 0 and expose_subdevices
        num_slots = props.numSubdevices if tiled else 1

        for slot in range(num_slots):
            if logical_index in visible:
                _cached_zes_device_infos.append(
                    _ZesDeviceInfo(
                        device_handle=device,
                        subdevice_id=slot if tiled else None,
                        is_integrated=is_integrated,
                    )
                )
                if is_integrated:
                    num_igpu += 1
                else:
                    num_dgpu += 1
            logical_index += 1

    # dGPUs take priority; strip iGPUs when at least one dGPU is visible.
    if num_dgpu and num_igpu:
        _cached_zes_device_infos = [
            info for info in _cached_zes_device_infos if not info.is_integrated
        ]
    return num_dgpu or num_igpu


def _raw_device_count_zes(visible_mask: list[int]) -> int:
    r"""Return the visible XPU device count via Level Zero Sysman, or negative on failure."""
    return _enum_zes_device_infos(visible_mask)


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
        # This test was protected via GIL. Double-check whether XPU has
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
    - ``last_level_cache_size`` (int): size in bytes of the device's last-level memory cache, shared across all Xe Cores (analogous to CUDA ``L2_cache_size``).
    - ``has_fp16`` (bool): whether float16 dtype is supported.
    - ``has_fp64`` (bool): whether float64 dtype is supported.
    - ``has_atomic64`` (bool): whether 64-bit atomic operations are supported.
    - ``has_bfloat16_conversions`` (bool): whether bfloat16 conversions are supported.
    - ``has_subgroup_matrix_multiply_accumulate`` (bool): whether DPAS (Dot Product Accumulate Systolic) is supported.
    - ``has_subgroup_matrix_multiply_accumulate_tensor_float32`` (bool): whether DPAS with tf32 inputs is supported.
    - ``has_subgroup_2d_block_io`` (bool): whether 2D block I/O for efficient matrix multiplication is supported.
    - ``is_integrated_gpu`` (bool): whether the device is an integrated GPU (iGPU).
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


def _import_pyzes():
    """Return the imported pyzes module; raise ImportError if missing, RuntimeError if the GPU driver is unavailable."""
    try:
        import pyzes  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "pyzes is required; install it with 'pip install pyzes'."
        ) from None
    except Exception as err:
        raise RuntimeError(
            "Failed to import pyzes. Ensure the GPU driver is installed with Level Zero Sysman support."
        ) from err
    return pyzes


def _get_pyzes_version() -> tuple[int, ...]:
    """
    Return the version of the pyzes package as a tuple of integers (major, minor, patch).
    Always ensure that the pyzes package is installed before calling this function.
    """
    from importlib.metadata import version

    return tuple(map(int, version("pyzes").split(".")))


def _zes_check(rc: int, msg: str) -> None:
    """Raise RuntimeError if the Level Zero Sysman call failed (rc != ZE_RESULT_SUCCESS)."""
    import pyzes  # type: ignore[import]

    if rc != pyzes.ZE_RESULT_SUCCESS:
        raise RuntimeError(f"{msg} (rc={rc})")


def _zes_ensure_device_infos(device: int):
    """Ensure the ZES device info cache is populated and validate the device index."""
    if not _cached_zes_device_infos:
        if _enum_zes_device_infos(_parse_visible_devices(strict=True)) < 0:
            raise RuntimeError("Failed to enumerate devices via Level Zero Sysman.")

    total_devices = len(_cached_zes_device_infos)
    if device >= total_devices:
        raise RuntimeError(
            f"The device {device} is out of range for Level Zero Sysman. It must be in the range [0, {total_devices})."
        )


def _get_zes_temperature_handle(device: Device = None) -> c_void_p:
    r"""Return the Level Zero Sysman GPU temperature sensor handle for the specified device.

    The result is cached in ``_ZesDeviceInfo.temperature_handle`` so that
    repeated calls skip sensor enumeration.  ``_cached_zes_device_infos``
    is lazily populated on the first call.

    Args:
        device (torch.device, str or int, optional): target device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).
    """
    pyzes = _import_pyzes()

    device = _get_device_index(device, optional=True)
    _zes_ensure_device_infos(device)

    info = _cached_zes_device_infos[device]
    if info.temperature_handle is not None:
        return info.temperature_handle

    device_handle = info.device_handle
    subdevice_id = info.subdevice_id

    # Note [telemetry handle selection]:
    # For tiled dGPUs, pick the handle whose subdeviceId matches.
    # For non-tiled devices, pick the root-level (non-subdevice) handle.
    temp_count = c_uint32(0)
    _zes_check(
        pyzes.zesDeviceEnumTemperatureSensors(device_handle, byref(temp_count), None),
        "Can't get Level Zero Sysman temperature sensor count.",
    )
    if temp_count.value == 0:
        raise RuntimeError("No Level Zero Sysman temperature sensors found.")
    temp_handles = (pyzes.zes_temp_handle_t * temp_count.value)()
    _zes_check(
        pyzes.zesDeviceEnumTemperatureSensors(
            device_handle, byref(temp_count), temp_handles
        ),
        "Can't get Level Zero Sysman temperature sensor handles.",
    )

    temperature_handle = None
    for temp_handle in temp_handles:
        temp_props = pyzes.zes_temp_properties_t()
        temp_props.stype = pyzes.ZES_STRUCTURE_TYPE_TEMP_PROPERTIES
        _zes_check(
            pyzes.zesTemperatureGetProperties(temp_handle, byref(temp_props)),
            "Can't get Level Zero Sysman temperature properties.",
        )
        if temp_props.type != pyzes.ZES_TEMP_SENSORS_GPU:
            continue
        if subdevice_id is not None:
            if temp_props.onSubdevice and temp_props.subdeviceId == subdevice_id:
                temperature_handle = temp_handle
                break
        else:
            if not temp_props.onSubdevice:
                temperature_handle = temp_handle
                break

    if temperature_handle is None:
        raise RuntimeError("No Level Zero Sysman GPU temperature handle found.")
    info.temperature_handle = temperature_handle
    return temperature_handle


def temperature(device: Device = None) -> float:
    r"""Return the GPU temperature in degrees Celsius.

    Args:
        device (torch.device, str or int, optional): selected device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).

    .. note:: This API may require elevated privileges (e.g. ``sudo``) to access GPU temperature information.
    """
    temperature_handle = _get_zes_temperature_handle(device)

    import pyzes  # type: ignore[import]

    temp = c_double(0.0)
    rc = pyzes.zesTemperatureGetState(temperature_handle, byref(temp))
    if rc == pyzes.ZE_RESULT_ERROR_NOT_AVAILABLE:
        raise RuntimeError(
            "GPU temperature querying is not available. Try running with elevated privileges (e.g. sudo)."
        )
    if rc != pyzes.ZE_RESULT_SUCCESS:
        raise RuntimeError(f"Can't get Level Zero Sysman GPU temperature (rc={rc}).")
    return temp.value


def _get_zes_frequency_handle(device: Device = None) -> c_void_p:
    r"""Return the Level Zero Sysman GPU frequency domain handle for the specified device.

    The result is cached in ``_ZesDeviceInfo.frequency_handle`` so that
    repeated calls skip domain enumeration.  ``_cached_zes_device_infos``
    is lazily populated on the first call.

    Args:
        device (torch.device, str or int, optional): target device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).
    """
    pyzes = _import_pyzes()

    device = _get_device_index(device, optional=True)
    _zes_ensure_device_infos(device)

    info = _cached_zes_device_infos[device]
    if info.frequency_handle is not None:
        return info.frequency_handle

    device_handle = info.device_handle

    # Enumerate all frequency domains under this device handle.
    freq_count = c_uint32(0)
    _zes_check(
        pyzes.zesDeviceEnumFrequencyDomains(device_handle, byref(freq_count), None),
        "Can't get Level Zero Sysman frequency domains count.",
    )
    if freq_count.value == 0:
        raise RuntimeError("No Level Zero Sysman frequency domains found.")
    freq_handles = (pyzes.zes_freq_handle_t * freq_count.value)()
    _zes_check(
        pyzes.zesDeviceEnumFrequencyDomains(
            device_handle, byref(freq_count), freq_handles
        ),
        "Can't get Level Zero Sysman frequency domain handles.",
    )

    # TODO: pyzes lacks zesFrequencyGetProperties, so we cannot filter by
    # subdevice or domain type. We assume index 0 (ZES_FREQ_DOMAIN_GPU)
    # is the GPU frequency domain.
    frequency_handle = freq_handles[0]
    info.frequency_handle = frequency_handle
    return frequency_handle


def clock_rate(device: Device = None) -> float:
    r"""Return the GPU clock rate in MHz.

    Args:
        device (torch.device, str or int, optional): selected device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).
    """
    frequency_handle = _get_zes_frequency_handle(device)

    import pyzes  # type: ignore[import]

    freq_state = pyzes.zes_freq_state_t()
    rc = pyzes.zesFrequencyGetState(frequency_handle, byref(freq_state))
    if rc != pyzes.ZE_RESULT_SUCCESS:
        raise RuntimeError(f"Can't get Level Zero Sysman GPU clock rate (rc={rc}).")
    return freq_state.actual


def _get_zes_power_handle(device: Device = None) -> c_void_p:
    r"""Return the Level Zero Sysman GPU power domain handle for the specified device.

    The result is cached in ``_ZesDeviceInfo.power_handle`` so that
    repeated calls skip domain enumeration.  ``_cached_zes_device_infos``
    is lazily populated on the first call.

    Args:
        device (torch.device, str or int, optional): target device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).
    """
    pyzes = _import_pyzes()

    device = _get_device_index(device, optional=True)
    _zes_ensure_device_infos(device)

    info = _cached_zes_device_infos[device]
    if info.power_handle is not None:
        return info.power_handle

    device_handle = info.device_handle

    # Enumerate all power domains under this device handle.
    power_count = c_uint32(0)
    _zes_check(
        pyzes.zesDeviceEnumPowerDomains(device_handle, byref(power_count), None),
        "Can't get Level Zero Sysman power domains count.",
    )
    if power_count.value == 0:
        raise RuntimeError("No Level Zero Sysman power domains found.")
    power_handles = (pyzes.zes_pwr_handle_t * power_count.value)()
    _zes_check(
        pyzes.zesDeviceEnumPowerDomains(
            device_handle, byref(power_count), power_handles
        ),
        "Can't get Level Zero Sysman power domain handles.",
    )

    # TODO: pyzes lacks zesPowerGetProperties, so we cannot filter by
    # subdevice or domain type. We assume index 0 (ZES_POWER_DOMAIN_CARD)
    # is the GPU card power domain.
    power_handle = power_handles[0]
    info.power_handle = power_handle
    return power_handle


def power_draw(device: Device = None) -> float:
    r"""Return the GPU card power draw in watts.

    The value is computed by dividing the energy delta by the time delta between
    two energy-counter reads separated by a 100ms sampling interval.

    Args:
        device (torch.device, str or int, optional): selected device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).

    .. note:: This function blocks for approximately 100ms per call due to the
        sampling interval required to compute an accurate power reading.

    .. note:: This API may require elevated privileges (e.g. ``sudo``) to access GPU power information.
    """
    power_handle = _get_zes_power_handle(device)

    import pyzes  # type: ignore[import]

    counter_start = pyzes.zes_power_energy_counter_t()
    rc = pyzes.zesPowerGetEnergyCounter(power_handle, byref(counter_start))
    if rc == pyzes.ZE_RESULT_ERROR_NOT_AVAILABLE:
        raise RuntimeError(
            "GPU power draw querying is not available. Try running with elevated privileges (e.g. sudo)."
        )
    if rc != pyzes.ZE_RESULT_SUCCESS:
        raise RuntimeError(f"Can't get Level Zero Sysman GPU power draw (rc={rc}).")

    import time

    time.sleep(_zes_sample_interval_ms / 1000.0)

    counter_end = pyzes.zes_power_energy_counter_t()
    _zes_check(
        pyzes.zesPowerGetEnergyCounter(power_handle, byref(counter_end)),
        "Can't get Level Zero Sysman GPU power energy counter.",
    )
    # energy is in microjoules, timestamp is in microseconds (per L0 Sysman spec).
    # microjoules / microseconds = watts, so the micro factors cancel.
    dt = counter_end.timestamp - counter_start.timestamp
    return (counter_end.energy - counter_start.energy) / dt


def _get_zes_engine_handle(device: Device = None) -> c_void_p:
    r"""Return the Level Zero Sysman GPU engine group handle for the specified device.

    The result is cached in ``_ZesDeviceInfo.engine_handle`` so that
    repeated calls skip group enumeration.  ``_cached_zes_device_infos``
    is lazily populated on the first call.

    Args:
        device (torch.device, str or int, optional): target device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).
    """
    pyzes = _import_pyzes()

    device = _get_device_index(device, optional=True)
    _zes_ensure_device_infos(device)

    info = _cached_zes_device_infos[device]
    if info.engine_handle is not None:
        return info.engine_handle

    device_handle = info.device_handle
    subdevice_id = info.subdevice_id

    # See Note [telemetry handle selection]
    engine_count = c_uint32(0)
    _zes_check(
        pyzes.zesDeviceEnumEngineGroups(device_handle, byref(engine_count), None),
        "Can't get Level Zero Sysman engine group count.",
    )
    # TODO: zesDeviceEnumEngineGroups does not return ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS on privilege errors;
    # instead it succeeds with count=0. Treat that as an error with a helpful hint about elevated privileges.
    if engine_count.value == 0:
        raise RuntimeError(
            "No Level Zero Sysman engine groups found. The GPU may not support engine monitoring, or try running with elevated privileges (e.g. sudo)."
        )
    engine_handles = (pyzes.zes_engine_handle_t * engine_count.value)()
    _zes_check(
        pyzes.zesDeviceEnumEngineGroups(
            device_handle, byref(engine_count), engine_handles
        ),
        "Can't get Level Zero Sysman engine group handles.",
    )

    engine_handle = None
    for eng_handle in engine_handles:
        eng_props = pyzes.zes_engine_properties_t()
        eng_props.stype = pyzes.ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES
        _zes_check(
            pyzes.zesEngineGetProperties(eng_handle, byref(eng_props)),
            "Can't get Level Zero Sysman engine properties.",
        )
        if eng_props.type != pyzes.ZES_ENGINE_GROUP_ALL:
            continue
        if subdevice_id is not None:
            if eng_props.onSubdevice and eng_props.subdeviceId == subdevice_id:
                engine_handle = eng_handle
                break
        else:
            if not eng_props.onSubdevice:
                engine_handle = eng_handle
                break

    if engine_handle is None:
        raise RuntimeError("No Level Zero Sysman GPU engine handle found.")
    info.engine_handle = engine_handle
    return engine_handle


def utilization(device: Device = None) -> float:
    r"""Return the GPU engine utilization as a percentage.

    The value is computed by dividing the active-time delta by the time delta
    between two engine-activity reads separated by a 100ms sampling interval.

    Args:
        device (torch.device, str or int, optional): selected device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).

    .. note:: This function blocks for approximately 100ms per call due to the
        sampling interval required to compute an accurate utilization reading.

    .. note:: This API may require elevated privileges (e.g. ``sudo``) to access GPU utilization information.
    """
    engine_handle = _get_zes_engine_handle(device)

    import pyzes  # type: ignore[import]

    stats_start = pyzes.zes_engine_stats_t()
    rc = pyzes.zesEngineGetActivity(engine_handle, byref(stats_start))
    if rc == pyzes.ZE_RESULT_ERROR_NOT_AVAILABLE:
        raise RuntimeError(
            "GPU utilization querying is not available. Try running with elevated privileges (e.g. sudo)."
        )
    if rc != pyzes.ZE_RESULT_SUCCESS:
        raise RuntimeError(
            f"Can't get Level Zero Sysman GPU engine activity (rc={rc})."
        )

    import time

    time.sleep(_zes_sample_interval_ms / 1000.0)

    stats_end = pyzes.zes_engine_stats_t()
    _zes_check(
        pyzes.zesEngineGetActivity(engine_handle, byref(stats_end)),
        "Can't get Level Zero Sysman GPU engine activity.",
    )
    # activeTime and timestamp are monotonic counters in microseconds.
    dt = stats_end.timestamp - stats_start.timestamp
    return (stats_end.activeTime - stats_start.activeTime) / dt * 100


def _zes_get_memory_handle(device: Device = None) -> c_void_p:
    r"""Return the Level Zero Sysman GPU memory module handle for the specified device.

    The result is cached in ``_ZesDeviceInfo.memory_handle`` so that
    repeated calls skip module enumeration.  ``_cached_zes_device_infos``
    is lazily populated on the first call.

    Args:
        device (torch.device, str or int, optional): target device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).
    """
    pyzes = _import_pyzes()

    device = _get_device_index(device, optional=True)
    _zes_ensure_device_infos(device)

    info = _cached_zes_device_infos[device]
    if info.memory_handle is not None:
        return info.memory_handle

    device_handle = info.device_handle
    subdevice_id = info.subdevice_id

    # See Note [telemetry handle selection]
    mem_count = c_uint32(0)
    _zes_check(
        pyzes.zesDeviceEnumMemoryModules(device_handle, byref(mem_count), None),
        "Can't get Level Zero Sysman memory module count.",
    )
    if mem_count.value == 0:
        raise RuntimeError("No Level Zero Sysman memory modules found.")
    memory_handles = (pyzes.zes_mem_handle_t * mem_count.value)()
    _zes_check(
        pyzes.zesDeviceEnumMemoryModules(
            device_handle, byref(mem_count), memory_handles
        ),
        "Can't get Level Zero Sysman memory module handles.",
    )

    memory_handle = None
    for mem_handle in memory_handles:
        mem_props = pyzes.zes_mem_properties_t()
        mem_props.stype = pyzes.ZES_STRUCTURE_TYPE_MEM_PROPERTIES
        _zes_check(
            pyzes.zesMemoryGetProperties(mem_handle, byref(mem_props)),
            "Can't get Level Zero Sysman memory properties.",
        )
        if mem_props.location != pyzes.ZES_MEM_LOC_DEVICE:
            continue
        if subdevice_id is not None:
            if mem_props.onSubdevice and mem_props.subdeviceId == subdevice_id:
                memory_handle = mem_handle
                break
        else:
            if not mem_props.onSubdevice:
                memory_handle = mem_handle
                break

    if memory_handle is None:
        raise RuntimeError("No Level Zero Sysman GPU memory handle found.")
    info.memory_handle = memory_handle
    return memory_handle


def memory_usage(device: Device = None) -> float:
    r"""Return the GPU memory bandwidth usage as a percentage.

    The value is computed by dividing the byte-transfer delta by the time delta
    between two bandwidth-counter reads separated by a 100ms sampling interval,
    then normalizing by the peak bandwidth reported by the hardware.

    Args:
        device (torch.device, str or int, optional): selected device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).

    .. note:: This function blocks for approximately 100ms per call due to the
        sampling interval required to compute an accurate bandwidth reading.

    .. note:: This API may require elevated privileges (e.g. ``sudo``) to access GPU memory bandwidth usage information.
    """
    memory_handle = _zes_get_memory_handle(device)

    import pyzes  # type: ignore[import]

    bandwidth_start = pyzes.zes_mem_bandwidth_t()
    rc = pyzes.zesMemoryGetBandwidth(memory_handle, byref(bandwidth_start))
    if rc == pyzes.ZE_RESULT_ERROR_NOT_AVAILABLE:
        raise RuntimeError(
            "GPU memory bandwidth usage querying is not available. Try running with elevated privileges (e.g. sudo)."
        )
    if rc != pyzes.ZE_RESULT_SUCCESS:
        raise RuntimeError(
            f"Can't get Level Zero Sysman GPU memory bandwidth (rc={rc})."
        )

    import time

    time.sleep(_zes_sample_interval_ms / 1000.0)

    bandwidth_end = pyzes.zes_mem_bandwidth_t()
    _zes_check(
        pyzes.zesMemoryGetBandwidth(memory_handle, byref(bandwidth_end)),
        "Can't get Level Zero Sysman GPU memory bandwidth.",
    )
    dt = bandwidth_end.timestamp - bandwidth_start.timestamp
    # readCounter and writeCounter are cumulative byte counts (per L0 Sysman spec).
    read_delta = bandwidth_end.readCounter - bandwidth_start.readCounter
    write_delta = bandwidth_end.writeCounter - bandwidth_start.writeCounter
    # maxBandwidth is in bytes/sec; dt is in microseconds; the 1e6 factor converts dt to seconds.
    return 1e6 * (read_delta + write_delta) / (bandwidth_end.maxBandwidth * dt) * 100


def device_memory_used(device: Device = None) -> int:
    r"""Return the current GPU used global (device) memory in bytes.

    Args:
        device (torch.device, str or int, optional): selected device. Uses the
            current device, given by :func:`~torch.xpu.current_device`,
            if ``None`` (default).

    .. note:: This API may require elevated privileges (e.g. ``sudo``) to access GPU memory usage information.
    """
    memory_handle = _zes_get_memory_handle(device)

    import pyzes  # type: ignore[import]

    mem_state = pyzes.zes_mem_state_t()
    rc = pyzes.zesMemoryGetState(memory_handle, byref(mem_state))
    if rc == pyzes.ZE_RESULT_ERROR_NOT_AVAILABLE:
        raise RuntimeError(
            "GPU memory usage querying is not available. Try running with elevated privileges (e.g. sudo)."
        )
    if rc != pyzes.ZE_RESULT_SUCCESS:
        raise RuntimeError(f"Can't get Level Zero Sysman GPU memory state (rc={rc}).")
    mem_props = pyzes.zes_mem_properties_t()
    mem_props.stype = pyzes.ZES_STRUCTURE_TYPE_MEM_PROPERTIES
    _zes_check(
        pyzes.zesMemoryGetProperties(memory_handle, byref(mem_props)),
        "Can't get Level Zero Sysman memory properties.",
    )
    # TODO: Some drivers report physicalSize as 0 on client GPUs; fall back to
    # the allocatable size as an approximation in that case.
    total = mem_props.physicalSize if mem_props.physicalSize != 0 else mem_state.size
    return total - mem_state.free


# import here to avoid circular import
from .memory import (
    change_current_allocator,
    empty_cache,
    get_per_process_memory_fraction,
    list_gpu_processes,
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
    "clock_rate",
    "current_device",
    "current_stream",
    "default_generators",
    "device",
    "device_of",
    "device_count",
    "device_memory_used",
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
    "list_gpu_processes",
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
    "memory_usage",
    "MemPool",
    "power_draw",
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
    "temperature",
    "utilization",
]
