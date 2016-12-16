import os
import weakref
from torch.multiprocessing.common import ExtendedInitPickler
import torch.cuda


class StorageRef(object):
    def __init__(self, ptr):
        self.cdata = ptr


_shared_cache = weakref.WeakValueDictionary()


def _fd_id(fd):
    stat = os.fstat(fd)
    return (stat.st_ino, stat.st_dev)


def _shm_object_handle(handle):
    if len(handle) == 3 or len(handle) == 5:
        return handle[1]
    else:
        return _fd_id(handle[0])


def _shared_serialize(self, use_fd):
    handle, storage_ref = self._share(use_fd, StorageRef)
    object_handle = _shm_object_handle(handle)
    _shared_cache[object_handle] = storage_ref
    self._shared_incref()
    return handle


def _shared_deserialize(cls, handle):
    object_handle = _shm_object_handle(handle)
    new_storage = None

    storage_ref = _shared_cache.get(object_handle)
    if storage_ref is not None:
        new_storage = cls._new_with_weak_ptr(storage_ref)
        if new_storage is not None and len(handle) == 5:
            # CUDA handles include an offset and size
            offset, size = handle[3:5]
            new_storage = new_storage._new_view(offset, size)

    if new_storage is None:
        if cls.is_cuda:
            torch.cuda._lazy_init()
        new_storage, storage_ref = cls._new_shared(handle, StorageRef)
        _shared_cache[object_handle] = storage_ref

    new_storage._shared_decref()
    return new_storage


def _save_shared_args(cls, args):
    storage = cls()
    storage._shared_args = args
    storage._tensor_users = set()
    return storage


def _open_shared_fd(self, fd_map):
    shared_args = (fd_map[self._shared_args[0]], self._shared_args[1])
    storage = _shared_deserialize(type(self), shared_args)
    self._set_cdata(storage._cdata)
    for t in self._tensor_users:
        t.set_(storage, t.storage_offset(), t.size(), t.stride())
    del self._shared_args
    del self._tensor_users


def reduce_storage(self, obj):
    if isinstance(self, ExtendedInitPickler) and not obj.is_cuda:
        handle = obj._shared_serialize(True)
        self.register_extended_init(obj)
        return (_save_shared_args, (type(obj), handle,))
    else:
        handle = obj._shared_serialize(False)
        return (_shared_deserialize, (type(obj), handle,))


def _init_storage_sharing():
    from torch.storage import _StorageBase
    _StorageBase._shared_serialize = _shared_serialize
    _StorageBase._open_shared_fd = _open_shared_fd
