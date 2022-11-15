r"""
This package adds support for CUDA tensor types, that implement the same
function as CPU tensors, but they utilize GPUs for computation.

It is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports CUDA.

:ref:`cuda-semantics` has more details about working with CUDA.
"""

import contextlib
import os
import torch
from torch.types import Device
import traceback
import warnings
import threading
from functools import lru_cache
from typing import Any, List, Optional, Set, Tuple, Union
from ._utils import _get_device_index, _dummy_type
from .._utils import classproperty
from .graphs import CUDAGraph, graph_pool_handle, graph, \
    make_graphed_callables, is_current_stream_capturing
from .streams import ExternalStream, Stream, Event
from .. import device as _device
import torch._C

try:
    from torch._C import _cudart  # type: ignore[attr-defined]
except ImportError:
    _cudart = None

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls = []  # don't invoke these until initialization occurs
_is_in_bad_fork = getattr(torch._C, "_cuda_isInBadFork", lambda: False)
_device_t = Union[_device, str, int, None]


class _LazySeedTracker:
    # Since seeding is memory-less, only track the latest seed.
    # Note: `manual_seed_all` followed by `manual_seed` overwrites
    # the seed on current device. We track the order of **latest**
    # calls between these two API.
    def __init__(self):
        self.manual_seed_all_cb = None
        self.manual_seed_cb = None
        self.call_order = []

    def queue_seed_all(self, cb, traceback):
        self.manual_seed_all_cb = (cb, traceback)
        # update seed_all to be latest
        self.call_order = [self.manual_seed_cb, self.manual_seed_all_cb]

    def queue_seed(self, cb, traceback):
        self.manual_seed_cb = (cb, traceback)
        # update seed to be latest
        self.call_order = [self.manual_seed_all_cb, self.manual_seed_cb]

    def get_calls(self) -> List:
        return self.call_order


_lazy_seed_tracker = _LazySeedTracker()

# Define dummy _CudaDeviceProperties type if PyTorch was compiled without CUDA
if hasattr(torch._C, '_CudaDeviceProperties'):
    _CudaDeviceProperties = torch._C._CudaDeviceProperties
else:
    _CudaDeviceProperties = _dummy_type('_CudaDeviceProperties')  # type: ignore[assignment, misc]

# Global variables dynamically populated by native code
has_magma: bool = False
has_half: bool = False
default_generators: Tuple[torch._C.Generator] = ()  # type: ignore[assignment]

def _is_compiled() -> bool:
    r"""Returns true if compile with CUDA support."""
    return hasattr(torch._C, '_cuda_getDeviceCount')

def _nvml_based_avail() -> bool:
    return os.getenv('PYTORCH_NVML_BASED_CUDA_CHECK') == '1'

def is_available() -> bool:
    r"""Returns a bool indicating if CUDA is currently available."""
    if not _is_compiled():
        return False
    if _nvml_based_avail():
        # The user has set an env variable to request this availability check that attempts to avoid fork poisoning by
        # using NVML at the cost of a weaker CUDA availability assessment. Note that if NVML discovery/initialization
        # fails, this assessment falls back to the default CUDA Runtime API assessment (`cudaGetDeviceCount`)
        return device_count() > 0
    else:
        # The default availability inspection never throws and returns 0 if the driver is missing or can't
        # be initialized. This uses the CUDA Runtime API `cudaGetDeviceCount` which in turn initializes the CUDA Driver
        # API via `cuInit`
        return torch._C._cuda_getDeviceCount() > 0


def is_bf16_supported():
    r"""Returns a bool indicating if the current CUDA/ROCm device supports dtype bfloat16"""
    # Check for ROCm, if true return true, no ROCM_VERSION check required,
    # since it is supported on AMD GPU archs.
    if torch.version.hip:
        return True

    cu_vers = torch.version.cuda
    if cu_vers is not None:
        cuda_maj_decide = int(cu_vers.split('.')[0]) >= 11
    else:
        cuda_maj_decide = False
    return torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8 and cuda_maj_decide

def _sleep(cycles):
    torch._C._cuda_sleep(cycles)


def _check_capability():
    incorrect_binary_warn = """
    Found GPU%d %s which requires CUDA_VERSION >= %d to
     work properly, but your PyTorch was compiled
     with CUDA_VERSION %d. Please install the correct PyTorch binary
     using instructions from https://pytorch.org
    """

    old_gpu_warn = """
    Found GPU%d %s which is of cuda capability %d.%d.
    PyTorch no longer supports this GPU because it is too old.
    The minimum cuda capability supported by this library is %d.%d.
    """

    if torch.version.cuda is not None:  # on ROCm we don't want this check
        CUDA_VERSION = torch._C._cuda_getCompiledVersion()
        for d in range(device_count()):
            capability = get_device_capability(d)
            major = capability[0]
            minor = capability[1]
            name = get_device_name(d)
            current_arch = major * 10 + minor
            min_arch = min((int(arch.split("_")[1]) for arch in torch.cuda.get_arch_list()), default=35)
            if current_arch < min_arch:
                warnings.warn(old_gpu_warn % (d, name, major, minor, min_arch // 10, min_arch % 10))
            elif CUDA_VERSION <= 9000 and major >= 7 and minor >= 5:
                warnings.warn(incorrect_binary_warn % (d, name, 10000, CUDA_VERSION))

def _check_cubins():
    incompatible_device_warn = """
{} with CUDA capability sm_{} is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities {}.
If you want to use the {} GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
"""
    if torch.version.cuda is None:  # on ROCm we don't want this check
        return
    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return
    supported_sm = [int(arch.split('_')[1]) for arch in arch_list if 'sm_' in arch]
    for idx in range(device_count()):
        cap_major, cap_minor = get_device_capability(idx)
        # NVIDIA GPU compute architectures are backward compatible within major version
        supported = any([sm // 10 == cap_major for sm in supported_sm])
        if not supported:
            device_name = get_device_name(idx)
            capability = cap_major * 10 + cap_minor
            warnings.warn(incompatible_device_warn.format(device_name, capability, " ".join(arch_list), device_name))


def is_initialized():
    r"""Returns whether PyTorch's CUDA state has been initialized."""
    return _initialized and not _is_in_bad_fork()


def _lazy_call(callable, **kwargs):
    if is_initialized():
        callable()
    else:
        # TODO(torch_deploy): this accesses linecache, which attempts to read the
        # file system to get traceback info. Patch linecache or do something
        # else here if this ends up being important.
        global _lazy_seed_tracker
        if kwargs.get("seed_all", False):
            _lazy_seed_tracker.queue_seed_all(callable, traceback.format_stack())
        elif kwargs.get("seed", False):
            _lazy_seed_tracker.queue_seed(callable, traceback.format_stack())
        else:
            # Don't store the actual traceback to avoid memory cycle
            _queued_calls.append((callable, traceback.format_stack()))

_lazy_call(_check_capability)
_lazy_call(_check_cubins)


class DeferredCudaCallError(Exception):
    pass

OutOfMemoryError = torch._C._OutOfMemoryError

def init():
    r"""Initialize PyTorch's CUDA state.  You may need to call
    this explicitly if you are interacting with PyTorch via
    its C API, as Python bindings for CUDA functionality will not
    be available until this initialization takes place.  Ordinary users
    should not need this, as all of PyTorch's CUDA methods
    automatically initialize CUDA state on-demand.

    Does nothing if the CUDA state is already initialized.
    """
    _lazy_init()


def _lazy_init():
    global _initialized, _queued_calls
    if is_initialized() or hasattr(_tls, 'is_initializing'):
        return
    with _initialization_lock:
        # We be double-checked locking, boys!  This is OK because
        # the above test was GIL protected anyway.  The inner test
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
                "Cannot re-initialize CUDA in forked subprocess. To use CUDA with "
                "multiprocessing, you must use the 'spawn' start method")
        if not hasattr(torch._C, '_cuda_getDeviceCount'):
            raise AssertionError("Torch not compiled with CUDA enabled")
        if _cudart is None:
            raise AssertionError(
                "libcudart functions unavailable. It looks like you have a broken build?")
        # This function throws if there's a driver initialization error, no GPUs
        # are found or any other error occurs
        if 'CUDA_MODULE_LOADING' not in os.environ:
            os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
        torch._C._cuda_init()
        # Some of the queued calls may reentrantly call _lazy_init();
        # we need to just return without initializing in that case.
        # However, we must not let any *other* threads in!
        _tls.is_initializing = True

        for calls in _lazy_seed_tracker.get_calls():
            if calls:
                _queued_calls.append(calls)

        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = (f"CUDA call failed lazily at initialization with error: {str(e)}\n\n"
                           f"CUDA call was originally invoked at:\n\n{orig_traceback}")
                    raise DeferredCudaCallError(msg) from e
        finally:
            delattr(_tls, 'is_initializing')
        _initialized = True


def cudart():
    _lazy_init()
    return _cudart


class cudaStatus(object):
    SUCCESS: int = 0
    ERROR_NOT_READY: int = 34

class CudaError(RuntimeError):
    def __init__(self, code: int) -> None:
        msg = _cudart.cudaGetErrorString(_cudart.cudaError(code))
        super(CudaError, self).__init__('{0} ({1})'.format(msg, code))


def check_error(res: int) -> None:
    if res != _cudart.cudaError.success:
        raise CudaError(res)


class device(object):
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        if self.idx == -1:
            return
        self.prev_idx = torch.cuda.current_device()
        if self.prev_idx != self.idx:
            torch.cuda.set_device(self.idx)
        if not torch.jit.is_scripting():
            _lazy_init()

    def __exit__(self, type: Any, value: Any, traceback: Any):
        if self.prev_idx != self.idx:
            torch.cuda.set_device(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GPU, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        idx = obj.get_device() if obj.is_cuda else -1
        super(device_of, self).__init__(idx)


def set_device(device: _device_t) -> None:
    r"""Sets the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``CUDA_VISIBLE_DEVICES`` environmental variable.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = _get_device_index(device)
    if device >= 0:
        torch._C._cuda_setDevice(device)


def get_device_name(device: Optional[_device_t] = None) -> str:
    r"""Gets the name of a device.

    Args:
        device (torch.device or int, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """
    return get_device_properties(device).name


def get_device_capability(device: Optional[_device_t] = None) -> Tuple[int, int]:
    r"""Gets the cuda capability of a device.

    Args:
        device (torch.device or int, optional): device for which to return the
            device capability. This function is a no-op if this argument is
            a negative integer. It uses the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        tuple(int, int): the major and minor cuda capability of the device
    """
    prop = get_device_properties(device)
    return prop.major, prop.minor


def get_device_properties(device: _device_t) -> _CudaDeviceProperties:
    r"""Gets the properties of a device.

    Args:
        device (torch.device or int or str): device for which to return the
            properties of the device.

    Returns:
        _CudaDeviceProperties: the properties of the device
    """
    _lazy_init()  # will define _get_device_properties
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return _get_device_properties(device)  # type: ignore[name-defined]

def can_device_access_peer(device: _device_t, peer_device: _device_t) -> bool:
    r"""Checks if peer access between two devices is possible.
    """
    _lazy_init()
    device = _get_device_index(device, optional=True)
    peer_device = _get_device_index(peer_device)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    if peer_device < 0 or peer_device >= device_count():
        raise AssertionError("Invalid peer device id")
    return torch._C._cuda_canDeviceAccessPeer(device, peer_device)


class StreamContext(object):
    r"""Context-manager that selects a given stream.

    All CUDA kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """
    cur_stream : Optional['torch.cuda.Stream']

    def __init__(self, stream: Optional['torch.cuda.Stream']):
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                self.idx = -1

        self.src_prev_stream = None if not torch.jit.is_scripting() else torch.cuda.default_stream(None)
        self.dst_prev_stream = None if not torch.jit.is_scripting() else torch.cuda.default_stream(None)

    def __enter__(self):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None or CUDA device not available
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = torch.cuda.current_stream(None)

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = torch.cuda.current_stream(cur_stream.device)
        torch.cuda.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no CUDA device available, return
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            torch.cuda.set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        torch.cuda.set_stream(self.src_prev_stream)  # type: ignore[arg-type]

def stream(stream: Optional['torch.cuda.Stream']) -> StreamContext:
    r"""Wrapper around the Context-manager StreamContext that
    selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    ..Note:: In eager mode stream is of type Stream class while in JIT it is
    an object of the custom class ``torch.classes.cuda.Stream``.
    """
    return StreamContext(stream)

def set_stream(stream: Stream):
    r"""Sets the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    torch._C._cuda_setStream(stream._cdata)

def _parse_visible_devices() -> Set[int]:
    """Parse CUDA_VISIBLE_DEVICES environment variable."""
    var = os.getenv("CUDA_VISIBLE_DEVICES")
    if var is None:
        return set(x for x in range(64))

    def _strtoul(s: str) -> int:
        """Return -1 or positive integer sequence string starts with,"""
        if not s:
            return -1
        for idx, c in enumerate(s):
            if not c.isdigit():
                break
            if idx + 1 == len(s):
                idx += 1
        return int(s[:idx]) if idx > 0 else -1

    # CUDA_VISIBLE_DEVICES uses something like strtoul
    # which makes `1gpu2,2ampere` is equivalent to `1,2`
    rc: Set[int] = set()
    for elem in var.split(","):
        rc.add(_strtoul(elem.strip()))
    return rc

def _raw_device_count_nvml() -> int:
    """Return number of devices as reported by NVML
    or negative value if NVML discovery/initialization failed."""
    from ctypes import CDLL, c_int
    nvml_h = CDLL("libnvidia-ml.so.1")
    rc = nvml_h.nvmlInit()
    if rc != 0:
        warnings.warn("Can't initialize NVML")
        return -1
    dev_arr = (c_int * 1)(-1)
    rc = nvml_h.nvmlDeviceGetCount_v2(dev_arr)
    if rc != 0:
        warnings.warn("Can't get nvml device count")
        return -1
    del nvml_h
    return dev_arr[0]

def _device_count_nvml() -> int:
    """Return number of devices as reported by NVML taking CUDA_VISIBLE_DEVICES into account.
    Negative value is returned if NVML discovery or initialization has failed."""
    visible_devices = _parse_visible_devices()
    if not visible_devices:
        return 0
    try:
        raw_cnt = _raw_device_count_nvml()
    except OSError:
        return -1
    except AttributeError:
        return -1
    if raw_cnt <= 0:
        return raw_cnt
    return len(set(range(raw_cnt)).intersection(visible_devices))

@lru_cache(maxsize=1)
def device_count() -> int:
    r"""Returns the number of GPUs available."""
    if not _is_compiled():
        return 0
    nvml_count = _device_count_nvml()
    return torch._C._cuda_getDeviceCount() if nvml_count < 0 else nvml_count

def get_arch_list() -> List[str]:
    r"""Returns list CUDA architectures this library was compiled for."""
    if not is_available():
        return []
    arch_flags = torch._C._cuda_getArchFlags()
    if arch_flags is None:
        return []
    return arch_flags.split()

def get_gencode_flags() -> str:
    r"""Returns NVCC gencode flags this library was compiled with."""
    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return ""
    arch_list_ = [arch.split("_") for arch in arch_list]
    return " ".join([f"-gencode compute=compute_{arch},code={kind}_{arch}" for (kind, arch) in arch_list_])



def current_device() -> int:
    r"""Returns the index of a currently selected device."""
    _lazy_init()
    return torch._C._cuda_getDevice()


def synchronize(device: _device_t = None) -> None:
    r"""Waits for all kernels in all streams on a CUDA device to complete.

    Args:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    _lazy_init()
    with torch.cuda.device(device):
        return torch._C._cuda_synchronize()


def ipc_collect():
    r"""Force collects GPU memory after it has been released by CUDA IPC.

    .. note::
        Checks if any sent CUDA tensors could be cleaned from the memory. Force
        closes shared memory file used for reference counting if there is no
        active counters. Useful when the producer process stopped actively sending
        tensors and want to release unused memory.
    """
    _lazy_init()
    return torch._C._cuda_ipc_collect()


def current_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Returns the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    return Stream(_cdata=torch._C._cuda_getCurrentStream(
        _get_device_index(device, optional=True)))


def default_stream(device: Optional[_device_t] = None) -> Stream:
    r"""Returns the default :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    return Stream(_cdata=torch._C._cuda_getDefaultStream(
        _get_device_index(device, optional=True)))


def current_blas_handle():
    r"""Returns cublasHandle_t pointer to current cuBLAS handle"""
    _lazy_init()
    return torch._C._cuda_getCurrentBlasHandle()

def set_sync_debug_mode(debug_mode: Union[int, str]) -> None:
    r"""Sets the debug mode for cuda synchronizing operations.

    Args:
        debug_mode(str or int): if "default" or 0, don't error or warn on synchronizing operations,
            if "warn" or 1, warn on synchronizing operations, if "error" or 2, error out synchronizing operations.

    Warning:
        This is an experimental feature, and not all synchronizing operations will trigger warning or error. In
        particular, operations in torch.distributed and torch.sparse namespaces are not covered yet.
    """

    _lazy_init()
    if isinstance(debug_mode, str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            raise RuntimeError("invalid value of debug_mode, expected one of `default`, `warn`, `error`")

    torch._C._cuda_set_sync_debug_mode(debug_mode)

def get_sync_debug_mode() -> int:
    r"""Returns current value of debug mode for cuda synchronizing operations."""

    _lazy_init()
    return torch._C._cuda_get_sync_debug_mode()


def memory_usage(device: Optional[Union[Device, int]] = None) -> int:
    r"""Returns the percent of time over the past sample period during which global (device)
    memory was being read or written. as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    try:
        import pynvml  # type: ignore[import]
    except ModuleNotFoundError:
        raise ModuleNotFoundError("pynvml module not found, please install pynvml")
    from pynvml import NVMLError_DriverNotLoaded
    try:
        pynvml.nvmlInit()
    except NVMLError_DriverNotLoaded:
        raise RuntimeError("cuda driver can't be loaded, is cuda enabled?")
    device = _get_device_index(device, optional=True)
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    return pynvml.nvmlDeviceGetUtilizationRates(handle).memory


def utilization(device: Optional[Union[Device, int]] = None) -> int:
    r"""Returns the percent of time over the past sample period during which one or
    more kernels was executing on the GPU as given by `nvidia-smi`.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    try:
        import pynvml  # type: ignore[import]
    except ModuleNotFoundError:
        raise ModuleNotFoundError("pynvml module not found, please install pynvml")
    from pynvml import NVMLError_DriverNotLoaded
    try:
        pynvml.nvmlInit()
    except NVMLError_DriverNotLoaded:
        raise RuntimeError("cuda driver can't be loaded, is cuda enabled?")
    device = _get_device_index(device, optional=True)
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu


from .memory import *  # noqa: F403


from .random import *  # noqa: F403

################################################################################
# Define Storage and Tensor classes
################################################################################

@staticmethod  # type: ignore[misc]
def _lazy_new(cls, *args, **kwargs):
    _lazy_init()
    # We may need to call lazy init again if we are a forked child
    # del _CudaBase.__new__
    return super(_CudaBase, cls).__new__(cls, *args, **kwargs)


class _CudaBase(object):
    is_cuda = True
    is_sparse = False

    def type(self, *args, **kwargs):
        # We could use a Protocol here to tell mypy that self has `get_device` method
        # but it is only available in the typing module on Python >= 3.8
        # or on typing_extensions module on Python >= 3.6
        with device(self.get_device()):  # type: ignore[attr-defined]
            return super(_CudaBase, self).type(*args, **kwargs)  # type: ignore[misc]

    __new__ = _lazy_new

from torch.storage import _LegacyStorage

class _CudaLegacyStorage(_LegacyStorage):
    @classmethod
    def from_buffer(cls, *args, **kwargs):
        raise RuntimeError('from_buffer: Not available for CUDA storage')

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs):
        raise RuntimeError('_new_with_weak_ptr: Not available for CUDA storage')

    @classmethod
    def _new_shared_filename(cls, manager, obj, size, *, device=None, dtype=None):
        raise RuntimeError('_new_shared_filename: Not available for CUDA storage')

class ByteStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.uint8

class DoubleStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.double

class FloatStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.float

class HalfStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.half

class LongStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.long

class IntStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.int

class ShortStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.short

class CharStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.int8

class BoolStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.bool

class BFloat16Storage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.bfloat16

class ComplexDoubleStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.cdouble

class ComplexFloatStorage(_CudaLegacyStorage):
    @classproperty
    def dtype(self):
        return torch.cfloat

del _LegacyStorage
del _CudaLegacyStorage

torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(BoolStorage)
torch._storage_classes.add(BFloat16Storage)
torch._storage_classes.add(ComplexDoubleStorage)
torch._storage_classes.add(ComplexFloatStorage)

from . import sparse
from . import profiler
from . import nvtx
from . import amp
from . import jiterator

__all__ = [
    # Typed storage and tensors
    'BFloat16Storage', 'BFloat16Tensor',
    'BoolStorage', 'BoolTensor',
    'ByteStorage', 'ByteTensor',
    'CharStorage', 'CharTensor',
    'ComplexDoubleStorage', 'ComplexFloatStorage',
    'DoubleStorage', 'DoubleTensor',
    'FloatStorage', 'FloatTensor',
    'HalfStorage', 'HalfTensor',
    'IntStorage', 'IntTensor',
    'LongStorage', 'LongTensor',
    'ShortStorage', 'ShortTensor',
    'CUDAGraph', 'CudaError', 'DeferredCudaCallError', 'Event', 'ExternalStream', 'OutOfMemoryError',
    'Stream', 'StreamContext', 'amp', 'caching_allocator_alloc', 'caching_allocator_delete', 'can_device_access_peer',
    'check_error', 'cudaStatus', 'cudart', 'current_blas_handle', 'current_device', 'current_stream', 'default_generators',
    'default_stream', 'device', 'device_count', 'device_of', 'empty_cache', 'get_allocator_backend', 'get_arch_list',
    'get_device_capability', 'get_device_name', 'get_device_properties', 'get_gencode_flags', 'get_rng_state', 'get_rng_state_all',
    'get_sync_debug_mode', 'graph', 'graph_pool_handle', 'graphs', 'has_half', 'has_magma', 'init', 'initial_seed', 'ipc_collect',
    'is_available', 'is_bf16_supported', 'is_current_stream_capturing', 'is_initialized', 'jiterator', 'list_gpu_processes',
    'make_graphed_callables', 'manual_seed', 'manual_seed_all', 'max_memory_allocated', 'max_memory_cached', 'max_memory_reserved',
    'mem_get_info', 'memory', 'memory_allocated', 'memory_cached', 'memory_reserved', 'memory_snapshot', 'memory_stats',
    'memory_stats_as_nested_dict', 'memory_summary', 'memory_usage', 'nccl', 'nvtx', 'profiler', 'random',
    'reset_accumulated_memory_stats', 'reset_max_memory_allocated', 'reset_max_memory_cached', 'reset_peak_memory_stats',
    'seed', 'seed_all', 'set_device', 'set_per_process_memory_fraction', 'set_rng_state', 'set_rng_state_all', 'set_stream',
    'set_sync_debug_mode', 'sparse', 'stream', 'streams', 'synchronize', 'utilization']
