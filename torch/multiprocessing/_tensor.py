import torch


def _shared_serialize(self):
    metadata = (self.storage_offset(), self.size(), self.stride())
    storage = self.storage()
    return (storage, metadata)


def _shared_deserialize(cls, args):
    storage, metadata = args
    storage_offset, size, stride = metadata
    new_tensor = cls()
    if hasattr(storage, '_tensor_users'):
        storage._tensor_users.add(new_tensor)
    new_tensor.set_(storage, storage_offset, size, stride)
    return new_tensor


def reduce_tensor(self, obj):
    return (_shared_deserialize, (type(obj), obj._shared_serialize(),))


def _init_tensor_sharing():
    from torch.tensor import _TensorBase
    _TensorBase._shared_serialize = _shared_serialize
