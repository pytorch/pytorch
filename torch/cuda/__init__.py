"""
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
from multiprocessing.util import register_after_fork as _register_after_fork

_initialized = False
_in_bad_fork = False
_original_pid = False
_cudart = None


def is_available():
    """Returns a bool indicating if CUDA is currently available."""
    if (not hasattr(torch._C, '_cuda_isDriverSufficient') or
            not torch._C._cuda_isDriverSufficient()):
        return False
    try:
        return torch._C._cuda_getDeviceCount() > 0
    except RuntimeError as e:
        if 'no CUDA-capable device is detected' in e.args[0]:
            return False
        raise


def _sleep(cycles):
    torch._C._cuda_sleep(cycles)


def _load_cudart():
    system = platform.system()
    lib_name = 'libcudart.' + ('dylib' if system == 'Darwin' else 'so')
    lib_paths = [
        lib_name,
        os.path.join(torch._C._cuda_getLibPath(), lib_name),
        os.path.join('/usr/local/cuda/lib64', lib_name),
        os.path.join('/usr/local/cuda/lib', lib_name),
    ]
    for path in lib_paths:
        try:
            return ctypes.cdll.LoadLibrary(path)
        except OSError:
            pass
    raise RuntimeError("couldn't find libcudart. Make sure CUDA libraries "
                       "are installed in a default location, or that they're in " +
                       ("DYLD_LIBRARY_PATH" if system == 'Darwin' else "LD_LIBRARY_PATH") +
                       ".")


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
Alternatively, go to: https://pytorch.org/binaries to install
a PyTorch version that has been compiled with your version
of the CUDA driver.""".format(str(torch._C._cuda_getDriverVersion())))


def _lazy_init():
    global _initialized, _cudart, _original_pid
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
    assert torch._C._cuda_init()
    _cudart = _load_cudart()
    _cudart.cudaGetErrorName.restype = ctypes.c_char_p
    _cudart.cudaGetErrorString.restype = ctypes.c_char_p
    _original_pid = os.getpid()
    _initialized = True


def _after_fork(arg):
    global _initialized, _in_bad_fork
    if _initialized and _original_pid != os.getpid():
        _initialized = False
        _in_bad_fork = True


_register_after_fork(_after_fork, _after_fork)


def cudart():
    _lazy_init()
    return _cudart


class device(object):
    """Context-manager that changes the selected device.

    Arguments:
        idx (int): device index to select. It's a no-op if this argument
            is negative.
    """

    def __init__(self, idx):
        self.idx = idx
        self.prev_idx = -1

    def __enter__(self):
        if self.idx is -1:
            return
        _lazy_init()
        self.prev_idx = torch._C._cuda_getDevice()
        if self.prev_idx != self.idx:
            torch._C._cuda_setDevice(self.idx)

    def __exit__(self, *args):
        if self.prev_idx != self.idx:
            torch._C._cuda_setDevice(self.prev_idx)
        return False


class device_of(device):
    """Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GPU, this is a no-op.

    Arguments:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        idx = obj.get_device() if obj.is_cuda else -1
        super(device_of, self).__init__(idx)


def set_device(device):
    """Sets the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``CUDA_VISIBLE_DEVICES`` environmental variable.

    Arguments:
        device (int): selected device. This function is a no-op if this
            argument is negative.
    """
    if device >= 0:
        torch._C._cuda_setDevice(device)


@contextlib.contextmanager
def stream(stream):
    """Context-manager that selects a given stream.

    All CUDA kernels queued within its context will be enqueued on a selected
    stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    """
    if stream is None:
        yield
        return
    prev_stream = current_stream()
    torch._C._cuda_setStream(stream._cdata)
    try:
        yield
    finally:
        torch._C._cuda_setStream(prev_stream._cdata)


def device_count():
    """Returns the number of GPUs available."""
    _lazy_init()
    return torch._C._cuda_getDeviceCount()


def current_device():
    """Returns the index of a currently selected device."""
    _lazy_init()
    return torch._C._cuda_getDevice()


def synchronize():
    """Waits for all kernels in all streams on current device to complete."""
    _lazy_init()
    return torch._C._cuda_synchronize()


def current_stream():
    """Returns a currently selected :class:`Stream`."""
    _lazy_init()
    return torch.cuda.Stream(_cdata=torch._C._cuda_getCurrentStream())


def _host_allocator():
    _lazy_init()
    return torch._C._cuda_cudaHostAllocator()


from .random import *

################################################################################
# Define Storage and Tensor classes
################################################################################


from ..tensor import _TensorBase
from ..storage import _StorageBase

if not hasattr(torch._C, 'CudaDoubleStorageBase'):
    # Define dummy base classes
    for t in ['Double', 'Float', 'Long', 'Int', 'Short', 'Char', 'Byte', 'Half']:
        storage_name = 'Cuda{0}StorageBase'.format(t)
        tensor_name = 'Cuda{0}TensorBase'.format(t)

        torch._C.__dict__[storage_name] = type(storage_name, (object,), {})
        torch._C.__dict__[tensor_name] = type(tensor_name, (object,), {})

    torch._C.__dict__['_CudaStreamBase'] = type('CudaStreamBase', (object,), {})


class _CudaBase(object):
    is_cuda = True

    def type(self, *args, **kwargs):
        with device(self.get_device()):
            return super(_CudaBase, self).type(*args, **kwargs)

    def __new__(cls, *args, **kwargs):
        _lazy_init()
        # We need this method only for lazy init, so we can remove it
        del _CudaBase.__new__
        return super(_CudaBase, cls).__new__(cls, *args, **kwargs)


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


class DoubleTensor(_CudaBase, torch._C.CudaDoubleTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return DoubleStorage


class FloatTensor(_CudaBase, torch._C.CudaFloatTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return FloatStorage


class LongTensor(_CudaBase, torch._C.CudaLongTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return LongStorage


class IntTensor(_CudaBase, torch._C.CudaIntTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return IntStorage


class ShortTensor(_CudaBase, torch._C.CudaShortTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return ShortStorage


class CharTensor(_CudaBase, torch._C.CudaCharTensorBase, _TensorBase):

    def is_signed(self):
        # TODO
        return False

    @classmethod
    def storage_type(cls):
        return CharStorage


class ByteTensor(_CudaBase, torch._C.CudaByteTensorBase, _TensorBase):

    def is_signed(self):
        return False

    @classmethod
    def storage_type(cls):
        return ByteStorage


class HalfTensor(_CudaBase, torch._C.CudaHalfTensorBase, _TensorBase):

    def is_signed(self):
        return True

    @classmethod
    def storage_type():
        return HalfStorage


torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)
torch._storage_classes.add(HalfStorage)

torch._tensor_classes.add(DoubleTensor)
torch._tensor_classes.add(FloatTensor)
torch._tensor_classes.add(LongTensor)
torch._tensor_classes.add(IntTensor)
torch._tensor_classes.add(ShortTensor)
torch._tensor_classes.add(CharTensor)
torch._tensor_classes.add(ByteTensor)
torch._tensor_classes.add(HalfTensor)

from .streams import Stream, Event
