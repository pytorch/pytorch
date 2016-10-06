from __future__ import print_function
import torch

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


class device(object):
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
            torch._C._cuda_setDevice(prev_idx)
        return False


class device_of(device):
    def __init__(self, tensor):
        idx = tensor.get_device() if tensor.is_cuda else -1
        super(device_of, self).__init__(idx)


def device_count():
    _lazy_init()
    return torch._C._cuda_getDeviceCount()


def current_device():
    _lazy_init()
    return torch._C._cuda_getDevice()


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


class _CudaBase(object):
    is_cuda = True

    def type(self, *args, **kwargs):
        with device(self.get_device()):
            return super(_CudaBase, self).type(*args, **kwargs)

    def new(self, *args, **kwargs):
        with device(kwargs.pop('device', self.get_device())):
            return super(_CudaBase, self).new(*args, **kwargs)

    def __new__(cls, *args, **kwargs):
        _lazy_init()
        with device(kwargs.pop('device', -1)):
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
class FloatTensor(_CudaBase, torch._C.CudaFloatTensorBase, _TensorBase):
    def is_signed(self):
        return True
class LongTensor(_CudaBase, torch._C.CudaLongTensorBase, _TensorBase):
    def is_signed(self):
        return True
class IntTensor(_CudaBase, torch._C.CudaIntTensorBase, _TensorBase):
    def is_signed(self):
        return True
class ShortTensor(_CudaBase, torch._C.CudaShortTensorBase, _TensorBase):
    def is_signed(self):
        return True
class CharTensor(_CudaBase, torch._C.CudaCharTensorBase, _TensorBase):
    def is_signed(self):
        # TODO
        return False
class ByteTensor(_CudaBase, torch._C.CudaByteTensorBase, _TensorBase):
    def is_signed(self):
        return False
class HalfTensor(_CudaBase, torch._C.CudaHalfTensorBase, _TensorBase):
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
