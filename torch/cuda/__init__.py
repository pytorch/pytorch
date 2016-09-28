from __future__ import print_function
import torch
import contextlib

try:
    if torch._C._cuda_isDriverSufficient() == False:
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
except AttributeError:
    raise ImportError("Torch not compiled with CUDA enabled")

@contextlib.contextmanager
def device(idx):
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
    return torch._C._cuda_getDeviceCount()


################################################################################
# Define Storage and Tensor classes
################################################################################


from .tensor import _CudaTensorBase
from .storage import _CudaStorageBase


class DoubleStorage(torch._C.CudaDoubleStorageBase, _CudaStorageBase):
    pass
class FloatStorage(torch._C.CudaFloatStorageBase, _CudaStorageBase):
    pass
class LongStorage(torch._C.CudaLongStorageBase, _CudaStorageBase):
    pass
class IntStorage(torch._C.CudaIntStorageBase, _CudaStorageBase):
    pass
class ShortStorage(torch._C.CudaShortStorageBase, _CudaStorageBase):
    pass
class CharStorage(torch._C.CudaCharStorageBase, _CudaStorageBase):
    pass
class ByteStorage(torch._C.CudaByteStorageBase, _CudaStorageBase):
    pass
class HalfStorage(torch._C.CudaHalfStorageBase, _CudaStorageBase):
    pass

class DoubleTensor(torch._C.CudaDoubleTensorBase, _CudaTensorBase):
    def is_signed(self):
        return True
class FloatTensor(torch._C.CudaFloatTensorBase, _CudaTensorBase):
    def is_signed(self):
        return True
class LongTensor(torch._C.CudaLongTensorBase, _CudaTensorBase):
    def is_signed(self):
        return True
class IntTensor(torch._C.CudaIntTensorBase, _CudaTensorBase):
    def is_signed(self):
        return True
class ShortTensor(torch._C.CudaShortTensorBase, _CudaTensorBase):
    def is_signed(self):
        return True
class CharTensor(torch._C.CudaCharTensorBase, _CudaTensorBase):
    def is_signed(self):
        # TODO
        return False
class ByteTensor(torch._C.CudaByteTensorBase, _CudaTensorBase):
    def is_signed(self):
        return False
class HalfTensor(torch._C.CudaHalfTensorBase, _CudaTensorBase):
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


def _cuda(self, idx=None, async=False):
    # This already is a CUDA tensor.
    # Let's check if it needs to be transfered to another GPU.
    if hasattr(self, 'get_device'):
        target_device = idx if idx else torch._C._cuda_getDevice()
        if self.get_device() != target_device:
            with device(target_device):
                return type(self)(self.size()).copy_(self, async)
        else:
            return self
    else:
        ctx = device(idx) if idx else _dummy_ctx()
        with ctx:
            return self.type(getattr(torch.cuda, self.__class__.__name__), async)


def _cpu(self):
    return self.type(getattr(torch, self.__class__.__name__))


from ..tensor import _TensorBase
from ..storage import _StorageBase
_TensorBase.cuda = _cuda
_TensorBase.cpu = _cpu
_StorageBase.cuda = _cuda
_StorageBase.cpu = _cpu


def current_device():
    return torch._C._cuda_getDevice()

assert torch._C._cuda_init()
