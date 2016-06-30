import torch

from torch.Storage import _StorageBase
from torch.Tensor import _TensorBase

################################################################################
# Define Storage and Tensor classes
################################################################################

class DoubleStorage(torch._C.CudaDoubleStorageBase, _StorageBase):
    pass
class FloatStorage(torch._C.CudaFloatStorageBase, _StorageBase):
    pass
class LongStorage(torch._C.CudaLongStorageBase, _StorageBase):
    pass
class IntStorage(torch._C.CudaIntStorageBase, _StorageBase):
    pass
class ShortStorage(torch._C.CudaShortStorageBase, _StorageBase):
    pass
class CharStorage(torch._C.CudaCharStorageBase, _StorageBase):
    pass
class ByteStorage(torch._C.CudaByteStorageBase, _StorageBase):
    pass
class HalfStorage(torch._C.CudaHalfStorageBase, _StorageBase):
    pass

class DoubleTensor(torch._C.CudaDoubleTensorBase, _TensorBase):
    pass
class FloatTensor(torch._C.CudaFloatTensorBase, _TensorBase):
    pass
class LongTensor(torch._C.CudaLongTensorBase, _TensorBase):
    pass
class IntTensor(torch._C.CudaIntTensorBase, _TensorBase):
    pass
class ShortTensor(torch._C.CudaShortTensorBase, _TensorBase):
    pass
class CharTensor(torch._C.CudaCharTensorBase, _TensorBase):
    pass
class ByteTensor(torch._C.CudaByteTensorBase, _TensorBase):
    pass
class HalfTensor(torch._C.CudaHalfTensorBase, _TensorBase):
    pass

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

assert torch._C._initCuda()
