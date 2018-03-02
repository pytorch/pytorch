import torch

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


class HalfStorage(_DistributedBase, torch._C.DistributedHalfStorageBase, _StorageBase):
    pass


torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)


_type_names = ['Double', 'Float', 'Half', 'Long', 'Int', 'Short', 'Char', 'Byte']
_locals = locals()
_tensors = [_locals[t + 'Tensor'] for t in _type_names]
_storages = [_locals[t + 'Storage'] for t in _type_names]
for cls in _tensors + _storages:
    cls.__module__ = 'torch.distributed'
torch._C._init_names(_tensors + _storages)
del _locals, _type_names, _tensors, _storages
