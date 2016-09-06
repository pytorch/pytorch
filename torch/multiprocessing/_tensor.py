import torch


def _shared_serialize(self):
    metadata = (self.storageOffset(), self.size().tolist(),
            self.stride().tolist())
    storage = self.storage()
    return (storage, metadata)


def _shared_deserialize(cls, args):
    storage, metadata = args
    storage_offset, size, stride = metadata
    size = torch.LongStorage(size)
    stride = torch.LongStorage(stride)
    new_tensor = cls()
    if hasattr(storage, '_tensor_users'):
        storage._tensor_users.add(new_tensor)
    new_tensor.set_(storage, storage_offset, size, stride)
    return new_tensor


def reduce_tensor(self, obj):
    return (_shared_deserialize, (type(obj), obj._shared_serialize(),))


def _init_tensor_sharing():
    from torch.Tensor import _TensorBase
    _TensorBase._shared_serialize = _shared_serialize

