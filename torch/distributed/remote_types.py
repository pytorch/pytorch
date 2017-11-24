import torch

from ..tensor import _TensorBase
from ..storage import _StorageBase


class _DistributedBase(object):
    is_cuda = False
    is_distributed = True


class DoubleStorage(_DistributedBase, torch._C.DistributedDoubleStorageBase, _StorageBase):
    pass


class FloatStorage(_DistributedBase, torch._C.DistributedFloatStorageBase, _StorageBase):
    pass


class LongStorage(_DistributedBase, torch._C.DistributedLongStorageBase, _StorageBase):
    pass


class IntStorage(_DistributedBase, torch._C.DistributedIntStorageBase, _StorageBase):
    pass


class ShortStorage(_DistributedBase, torch._C.DistributedShortStorageBase, _StorageBase):
    pass


class CharStorage(_DistributedBase, torch._C.DistributedCharStorageBase, _StorageBase):
    pass


class ByteStorage(_DistributedBase, torch._C.DistributedByteStorageBase, _StorageBase):
    pass


# class HalfStorage(_DistributedBase, torch._C.DistributedHalfStorageBase, _StorageBase):
    # pass


class DoubleTensor(_DistributedBase, torch._C.DistributedDoubleTensorBase, _TensorBase):
    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return DoubleStorage


class FloatTensor(_DistributedBase, torch._C.DistributedFloatTensorBase, _TensorBase):
    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return FloatStorage


class LongTensor(_DistributedBase, torch._C.DistributedLongTensorBase, _TensorBase):
    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return LongStorage


class IntTensor(_DistributedBase, torch._C.DistributedIntTensorBase, _TensorBase):
    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return IntStorage


class ShortTensor(_DistributedBase, torch._C.DistributedShortTensorBase, _TensorBase):
    def is_signed(self):
        return True

    @classmethod
    def storage_type(cls):
        return ShortStorage


class CharTensor(_DistributedBase, torch._C.DistributedCharTensorBase, _TensorBase):
    def is_signed(self):
        # TODO
        return False

    @classmethod
    def storage_type(cls):
        return CharStorage


class ByteTensor(_DistributedBase, torch._C.DistributedByteTensorBase, _TensorBase):
    def is_signed(self):
        return False

    @classmethod
    def storage_type(cls):
        return ByteStorage


# class HalfTensor(_DistributedBase, torch._C.DistributedHalfTensorBase, _TensorBase):
    # def is_signed(self):
        # return True
    # @classmethod
    # def storage_type():
        # return HalfStorage


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

torch._integer_tensor_classes.add(LongTensor)
torch._integer_tensor_classes.add(IntTensor)
torch._integer_tensor_classes.add(ShortTensor)
torch._integer_tensor_classes.add(CharTensor)
torch._integer_tensor_classes.add(ByteTensor)

_type_names = ['Double', 'Float', 'Long', 'Int', 'Short', 'Char', 'Byte']
_locals = locals()
_tensors = [_locals[t + 'Tensor'] for t in _type_names]
_storages = [_locals[t + 'Storage'] for t in _type_names]
for cls in _tensors + _storages:
    cls.__module__ = 'torch.distributed'
torch._C._init_names(_tensors + _storages)
del _locals, _type_names, _tensors, _storages
