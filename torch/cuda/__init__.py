from __future__ import print_function
import torch
import contextlib

_initialized = False

def is_available():
    return (hasattr(torch._C, '_cuda_isDriverSufficient') and
            torch._C._cuda_isDriverSufficient())

def _lazy_init():
    global _initialized
    if _initialized:
        return
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
    assert torch._C._cuda_init()
    _initialized = True


@contextlib.contextmanager
def device(idx):
    _lazy_init()
    prev_idx = torch._C._cuda_getDevice()
    if prev_idx != idx:
        torch._C._cuda_setDevice(idx)
        yield
        torch._C._cuda_setDevice(prev_idx)
    else:
        yield


@contextlib.contextmanager
def device_of(tensor):
    if tensor.is_cuda:
        with device(tensor.get_device()):
            yield
    else:
        yield


@contextlib.contextmanager
def _dummy_ctx():
    yield


def device_count():
    _lazy_init()
    return torch._C._cuda_getDeviceCount()

def current_device():
    _lazy_init()
    return torch._C._cuda_getDevice()

################################################################################
# Define Storage and Tensor classes
################################################################################


from .tensor import _CudaTensorBase
from .storage import _CudaStorageBase

if not hasattr(torch._C, 'CudaDoubleStorageBase'):
    # Define dummy base classes
    for t in ['Double', 'Float', 'Long', 'Int', 'Short', 'Char', 'Byte', 'Half']:
        storage_name = 'Cuda{0}StorageBase'.format(t)
        tensor_name = 'Cuda{0}TensorBase'.format(t)

        torch._C.__dict__[storage_name] = type(storage_name, (object,), {})
        torch._C.__dict__[tensor_name] = type(tensor_name, (object,), {})

class InitCuda(object):
    def __new__(cls, *args, **kwargs):
        _lazy_init()
        return super(InitCuda, cls).__new__(cls, *args, **kwargs)

class DoubleStorage(InitCuda, torch._C.CudaDoubleStorageBase, _CudaStorageBase):
    pass
class FloatStorage(InitCuda, torch._C.CudaFloatStorageBase, _CudaStorageBase):
    pass
class LongStorage(InitCuda, torch._C.CudaLongStorageBase, _CudaStorageBase):
    pass
class IntStorage(InitCuda, torch._C.CudaIntStorageBase, _CudaStorageBase):
    pass
class ShortStorage(InitCuda, torch._C.CudaShortStorageBase, _CudaStorageBase):
    pass
class CharStorage(InitCuda, torch._C.CudaCharStorageBase, _CudaStorageBase):
    pass
class ByteStorage(InitCuda, torch._C.CudaByteStorageBase, _CudaStorageBase):
    pass
class HalfStorage(InitCuda, torch._C.CudaHalfStorageBase, _CudaStorageBase):
    pass

class DoubleTensor(InitCuda, torch._C.CudaDoubleTensorBase, _CudaTensorBase):
    def is_signed(self):
        return True
class FloatTensor(InitCuda, torch._C.CudaFloatTensorBase, _CudaTensorBase):
    def is_signed(self):
        return True
class LongTensor(InitCuda, torch._C.CudaLongTensorBase, _CudaTensorBase):
    def is_signed(self):
        return True
class IntTensor(InitCuda, torch._C.CudaIntTensorBase, _CudaTensorBase):
    def is_signed(self):
        return True
class ShortTensor(InitCuda, torch._C.CudaShortTensorBase, _CudaTensorBase):
    def is_signed(self):
        return True
class CharTensor(InitCuda, torch._C.CudaCharTensorBase, _CudaTensorBase):
    def is_signed(self):
        # TODO
        return False
class ByteTensor(InitCuda, torch._C.CudaByteTensorBase, _CudaTensorBase):
    def is_signed(self):
        return False
class HalfTensor(InitCuda, torch._C.CudaHalfTensorBase, _CudaTensorBase):
    def is_signed(self):
        return True


torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)

torch._tensor_classes.add(DoubleTensor)
torch._tensor_classes.add(FloatTensor)
torch._tensor_classes.add(LongTensor)
torch._tensor_classes.add(IntTensor)
torch._tensor_classes.add(ShortTensor)
torch._tensor_classes.add(CharTensor)
torch._tensor_classes.add(ByteTensor)
