r"""
This package adds support for CUDA tensor types, that implement the same
function as CPU tensors, but they utilize GPUs for computation.

It is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports CUDA.

:ref:`cuda-semantics` has more details about working with CUDA.
"""

import contextlib
import platform
import ctypes
import os
import torch
import traceback
import warnings
from torch._six import raise_from
from subprocess import Popen, PIPE
from multiprocessing.util import register_after_fork as _register_after_fork
from ._utils import _get_device_index

_initialized = False
_queued_calls = []  # don't invoke these until initialization occurs
_in_bad_fork = False  # this global is also used in torch.manual_seed
_original_pid = False
_cudart = None


def find_cuda_windows_lib():
    proc = Popen(['where', 'cudart64*.dll'], stdout=PIPE, stderr=PIPE, stdin=PIPE)
    out, err = proc.communicate()
    out = out.decode().strip()
    if len(out) > 0:
        if out.find('\r\n') != -1:
            out = out.split('\r\n')[0]
        cuda_lib_name = os.path.basename(out)
        cuda_lib = os.path.splitext(cuda_lib_name)[0]
        cuda_lib = str(cuda_lib)
        return ctypes.cdll.LoadLibrary(cuda_lib)
    else:
        return None


def is_available():
    r"""Returns a bool indicating if CUDA is currently available."""
    if (not hasattr(torch._C, '_cuda_isDriverSufficient') or
            not torch._C._cuda_isDriverSufficient()):
        return False
    return torch._C._cuda_getDeviceCount() > 0


def _sleep(cycles):
    torch._C._cuda_sleep(cycles)


def _load_cudart():
    # First check the main program for CUDA symbols
    if platform.system() == 'Windows':
        lib = find_cuda_windows_lib()
    else:
        lib = ctypes.cdll.LoadLibrary(None)
    if hasattr(lib, 'cudaGetErrorName'):
        return lib

    raise RuntimeError(
        "couldn't find libcudart. Make sure CUDA libraries are installed in a "
        "default location, or that they're in {}."
        .format('DYLD_LIBRARY_PATH' if platform.system() == 'Darwin' else
                'LD_LIBRARY_PATH'))


def _check_driver():
    if not hasattr(torch._C, '_cuda_isDriverSufficient'):
        raise AssertionError("Torch not compiled with CUDA enabled")
    if not torch._C._cuda_isDriverSufficient():
        if torch._C._cuda_getDriverVersion() == 0:
            # found no NVIDIA driver on the system
            raise AssertionError("""
Found no NVIDIA driver on your system. Please check that you
have an NVIDIA GPU and installed a driver from
http://www.nvidia.com/Download/index.aspx""")
        else:
            # TODO: directly link to the alternative bin that needs install
            raise AssertionError("""
The NVIDIA driver on your system is too old (found version {}).
Please update your GPU driver by downloading and installing a new
version from the URL: http://www.nvidia.com/Download/index.aspx
Alternatively, go to: https://pytorch.org to install
a PyTorch version that has been compiled with your version
of the CUDA driver.""".format(str(torch._C._cuda_getDriverVersion())))


def _check_capability():
    incorrect_binary_warn = """
    Found GPU%d %s which requires CUDA_VERSION >= %d for
     optimal performance and fast startup time, but your PyTorch was compiled
     with CUDA_VERSION %d. Please install the correct PyTorch binary
     using instructions from https://pytorch.org
    """

    old_gpu_warn = """
    Found GPU%d %s which is of cuda capability %d.%d.
    PyTorch no longer supports this GPU because it is too old.
    The minimum cuda capability that we support is 3.5.
    """

    CUDA_VERSION = torch._C._cuda_getCompiledVersion()
    for d in range(device_count()):
        capability = get_device_capability(d)
        major = capability[0]
        name = get_device_name(d)
        if CUDA_VERSION < 8000 and major >= 6:
            warnings.warn(incorrect_binary_warn % (d, name, 8000, CUDA_VERSION))
        elif CUDA_VERSION < 9000 and major >= 7:
            warnings.warn(incorrect_binary_warn % (d, name, 9000, CUDA_VERSION))
        elif capability == (3, 0) or major < 3:
            warnings.warn(old_gpu_warn % (d, name, major, capability[1]))


def _lazy_call(callable):
    if _initialized:
        callable()
    else:
        # Don't store the actual traceback to avoid memory cycle
        _queued_calls.append((callable, traceback.format_stack()))

_lazy_call(_check_capability)


class DeferredCudaCallError(Exception):
    pass


def init():
    r"""Initialize PyTorch's CUDA state.  You may need to call
    this explicitly if you are interacting with PyTorch via
    its C API, as Python bindings for CUDA functionality will not
    be until this initialization takes place.  Ordinary users
    should not need this, as all of PyTorch's CUDA methods
    automatically initialize CUDA state on-demand.

    Does nothing if the CUDA state is already initialized.
    """
    _lazy_init()


def _lazy_init():
    global _initialized, _cudart, _original_pid, _queued_calls
    if _initialized:
        return
    if _in_bad_fork:
        from sys import version_info
        if version_info < (3, 4):
            msg = ("To use CUDA with multiprocessing, you must use Python "
                   "3.4+ and the 'spawn' start method")
        else:
            msg = ("To use CUDA with multiprocessing, you must use the "
                   "'spawn' start method")
        raise RuntimeError(
            "Cannot re-initialize CUDA in forked subprocess. " + msg)
    _check_driver()
    torch._C._cuda_init()
    _cudart = _load_cudart()
    _cudart.cudaGetErrorName.restype = ctypes.c_char_p
    _cudart.cudaGetErrorString.restype = ctypes.c_char_p
    _original_pid = os.getpid()
    _initialized = True
    # Important to do this after _initialized, since some queued calls
    # may themselves call _lazy_init()
    for queued_call, orig_traceback in _queued_calls:
        try:
            queued_call()
        except Exception as e:
            msg = ("CUDA call failed lazily at initialization with error: {}\n\n"
                   "CUDA call was originally invoked at:\n\n{}").format(str(e), orig_traceback)
            raise_from(DeferredCudaCallError(msg), e)


def _after_fork(arg):
    global _initialized, _in_bad_fork
    if _initialized and _original_pid != os.getpid():
        _initialized = False
        _in_bad_fork = True
        _CudaBase.__new__ = _lazy_new


_register_after_fork(_after_fork, _after_fork)


def cudart():
    _lazy_init()
    return _cudart


class cudaStatus(object):
    SUCCESS = 0
    ERROR_NOT_READY = 34


class CudaError(RuntimeError):
    def __init__(self, code):
        msg = cudart().cudaGetErrorString(code).decode('utf-8')
        super(CudaError, self).__init__('{0} ({1})'.format(msg, code))


def check_error(res):
    if res != cudaStatus.SUCCESS:
        raise CudaError(res)


class device(object):
    r"""Context-manager that changes the selected device.

    Arguments:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        if self.idx == -1:
            return
        self.prev_idx = torch._C._cuda_getDevice()
        if self.prev_idx != self.idx:
            torch._C._cuda_setDevice(self.idx)
        _lazy_init()

    def __exit__(self, *args):
        if self.prev_idx != self.idx:
            torch._C._cuda_setDevice(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GPU, this is a no-op.

    Arguments:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        idx = obj.get_device() if obj.is_cuda else -1
        super(device_of, self).__init__(idx)


def set_device(device):
    r"""Sets the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``CUDA_VISIBLE_DEVICES`` environmental variable.

    Arguments:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = _get_device_index(device)
    if device >= 0:
        torch._C._cuda_setDevice(device)


def get_device_name(device=None):
    r"""Gets the name of a device.

    Arguments:
        device (torch.device or int, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    return get_device_properties(device).name


def get_device_capability(device=None):
    r"""Gets the cuda capability of a device.

    Arguments:
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


def get_device_properties(device):
    if not _initialized:
        init()  # will define _get_device_properties and _CudaDeviceProperties
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return _get_device_properties(device)


@contextlib.contextmanager
def stream(stream):
    r"""Context-manager that selects a given stream.

    All CUDA kernels queued within its context will be enqueued on a selected
    stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.

    .. note:: Streams are per-device. If the selected stream is not on the
        current device, this function will also change the current device to
        match the stream.
    """
    if stream is None:
        yield
        return
    src_prev_stream = current_stream()

    if src_prev_stream.device != stream.device:
        # The given stream is on a different device; have to restore the
        # current_stream on that device on exit as well
        with device(stream.device):
            dst_prev_stream = current_stream()

    torch._C._cuda_setStream(stream._cdata)
    try:
        yield
    finally:
        if src_prev_stream.device != stream.device:
            torch._C._cuda_setStream(dst_prev_stream._cdata)
        torch._C._cuda_setStream(src_prev_stream._cdata)


def device_count():
    r"""Returns the number of GPUs available."""
    if is_available():
        return torch._C._cuda_getDeviceCount()
    else:
        return 0


def current_device():
    r"""Returns the index of a currently selected device."""
    _lazy_init()
    return torch._C._cuda_getDevice()


def synchronize(device=None):
    r"""Waits for all kernels in all streams on a CUDA device to complete.

    Arguments:
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


def current_stream(device=None):
    r"""Returns the currently selected :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    return torch.cuda.Stream(_cdata=torch._C._cuda_getCurrentStream(
        _get_device_index(device, optional=True)))


def default_stream(device=None):
    r"""Returns the default :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    return torch.cuda.Stream(_cdata=torch._C._cuda_getDefaultStream(
        _get_device_index(device, optional=True)))


def current_blas_handle():
    r"""Returns cublasHandle_t pointer to current cuBLAS handle"""
    _lazy_init()
    return torch._C._cuda_getCurrentBlasHandle()


def empty_cache():
    r"""Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU application and visible in
    `nvidia-smi`.

    .. note::
        :func:`~torch.cuda.empty_cache` doesn't increase the amount of GPU
        memory available for PyTorch. See :ref:`cuda-memory-management` for
        more details about GPU memory management.
    """
    if _initialized:
        torch._C._cuda_emptyCache()


def memory_allocated(device=None):
    r"""Returns the current GPU memory occupied by tensors in bytes for a given
    device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This is likely less than the amount shown in `nvidia-smi` since some
        unused memory can be held by the caching allocator and some context
        needs to be created on GPU. See :ref:`cuda-memory-management` for more
        details about GPU memory management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._cuda_memoryAllocated(device)


def max_memory_allocated(device=None):
    r"""Returns the maximum GPU memory occupied by tensors in bytes for a given
    device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.cuda.reset_max_memory_allocated` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._cuda_maxMemoryAllocated(device)


def reset_max_memory_allocated(device=None):
    r"""Resets the starting point in tracking maximum GPU memory occupied by
    tensors for a given device.

    See :func:`~torch.cuda.max_memory_allocated` for details.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._cuda_resetMaxMemoryAllocated(device)


def memory_cached(device=None):
    r"""Returns the current GPU memory managed by the caching allocator in bytes
    for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._cuda_memoryCached(device)


def max_memory_cached(device=None):
    r"""Returns the maximum GPU memory managed by the caching allocator in bytes
    for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.cuda.reset_max_memory_cached` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._cuda_maxMemoryCached(device)


def reset_max_memory_cached(device=None):
    r"""Resets the starting point in tracking maximum GPU memory managed by the
    caching allocator for a given device.

    See :func:`~torch.cuda.max_memory_cached` for details.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch._C._cuda_resetMaxMemoryCached(device)


def _host_allocator():
    _lazy_init()
    return torch._C._cuda_cudaHostAllocator()


@contextlib.contextmanager
def _free_mutex():
    torch._C._cuda_lock_mutex()
    try:
        yield
    finally:
        torch._C._cuda_unlock_mutex()


from .random import *

################################################################################
# Define Storage and Tensor classes
################################################################################


from ..storage import _StorageBase


def _dummy_type(name):
    def init_err(self):
        class_name = self.__class__.__name__
        raise RuntimeError(
            "Tried to instantiate dummy base class {}".format(class_name))
    return type(storage_name, (object,), {"__init__": init_err})


if not hasattr(torch._C, 'CudaDoubleStorageBase'):
    # Define dummy base classes
    for t in ['Double', 'Float', 'Long', 'Int', 'Short', 'Char', 'Byte', 'Half', 'Bool']:
        storage_name = 'Cuda{0}StorageBase'.format(t)
        tensor_name = 'Cuda{0}TensorBase'.format(t)

        torch._C.__dict__[storage_name] = _dummy_type(storage_name)
        torch._C.__dict__[tensor_name] = _dummy_type(tensor_name)

    torch._C.__dict__['_CudaStreamBase'] = _dummy_type('CudaStreamBase')
    torch._C.__dict__['_CudaEventBase'] = _dummy_type('CudaEventBase')


@staticmethod
def _lazy_new(cls, *args, **kwargs):
    _lazy_init()
    # We need this method only for lazy init, so we can remove it
    del _CudaBase.__new__
    return super(_CudaBase, cls).__new__(cls, *args, **kwargs)


class _CudaBase(object):
    is_cuda = True
    is_sparse = False

    def type(self, *args, **kwargs):
        with device(self.get_device()):
            return super(_CudaBase, self).type(*args, **kwargs)

    __new__ = _lazy_new


class DoubleStorage(_CudaBase, torch._C.CudaDoubleStorageBase, _StorageBase):
    pass


class FloatStorage(_CudaBase, torch._C.CudaFloatStorageBase, _StorageBase):
    pass


class LongStorage(_CudaBase, torch._C.CudaLongStorageBase, _StorageBase):
    pass


class IntStorage(_CudaBase, torch._C.CudaIntStorageBase, _StorageBase):
    pass


class ShortStorage(_CudaBase, torch._C.CudaShortStorageBase, _StorageBase):
    pass


class CharStorage(_CudaBase, torch._C.CudaCharStorageBase, _StorageBase):
    pass


class ByteStorage(_CudaBase, torch._C.CudaByteStorageBase, _StorageBase):
    pass


class HalfStorage(_CudaBase, torch._C.CudaHalfStorageBase, _StorageBase):
    pass


class BoolStorage(_CudaBase, torch._C.CudaBoolStorageBase, _StorageBase):
    pass

torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(BoolStorage)

from . import sparse  # noqa: F401
from . import profiler  # noqa: F401
from . import nvtx  # noqa: F401
from .streams import Stream, Event  # noqa: F401
