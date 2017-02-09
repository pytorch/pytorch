import torch
import os
import weakref
import multiprocessing
from multiprocessing.reduction import ForkingPickler
import sys
try:
    # Early load resource_sharer to prevent a partially initialized instance
    # from being inherited in a forked child process. The reduce_storage method
    # requires this module indirectly through DupFd(). The built-in mp.Queue
    # class pickles arguments in a background thread which may overlap with the
    # fork.
    import multiprocessing.resource_sharer
except ImportError:
    pass


class StorageRef(object):
    # An object with a cdata field which may be set to None. We subclass object
    # instead of using a dict() to support weak references.

    def __init__(self, ptr):
        self.cdata = ptr


# mapping from handles to StorageRef objects
shared_cache = weakref.WeakValueDictionary()


def rebuild_event(handle):
    return torch.cuda.Event(_handle=handle)


def reduce_event(event):
    return (rebuild_event, (event.ipc_handle(),))


def rebuild_tensor(cls, storage, metadata):
    storage_offset, size, stride = metadata
    new_tensor = cls()
    new_tensor.set_(storage, storage_offset, size, stride)
    return new_tensor


def reduce_tensor(tensor):
    metadata = (tensor.storage_offset(), tensor.size(), tensor.stride())
    storage = tensor.storage()
    return (rebuild_tensor, (type(tensor), storage, metadata))


def fd_id(fd):
    # Returns a tuple which uniquely identifies a file descriptor. In Mac OS,
    # this doesn't work with shared memory handles, which is why we don't
    # support the "file_descriptor" sharing method on that platform.
    stat = os.fstat(fd)
    return (stat.st_ino, stat.st_dev)


def storage_from_cache(cls, key):
    storage_ref = shared_cache.get(key)
    if storage_ref is None:
        return None
    return cls._new_with_weak_ptr(storage_ref)


def rebuild_storage_fd(cls, df, size):
    if sys.version_info[0] == 2:
        fd = multiprocessing.reduction.rebuild_handle(df)
    else:
        fd = df.detach()
    try:
        storage = storage_from_cache(cls, fd_id(fd))
        if storage is not None:
            return storage
        storage = cls._new_shared_fd(fd, size)
        shared_cache[fd_id(fd)] = storage._weak_ref(StorageRef)
        return storage
    finally:
        os.close(fd)


def rebuild_storage_filename(cls, manager, handle, size):
    storage = storage_from_cache(cls, handle)
    if storage is not None:
        return storage._shared_decref()
    storage = cls._new_shared_filename(manager, handle, size)
    shared_cache[handle] = storage._weak_ref(StorageRef)
    return storage._shared_decref()


def rebuild_storage_cuda(cls, device, handle, size, offset, view_size):
    storage = storage_from_cache(cls, handle)
    if storage is not None:
        return storage._new_view(offset, view_size)
    torch.cuda._lazy_init()
    storage = cls._new_shared_cuda(device, handle, size, offset, view_size)
    shared_cache[handle] = storage._weak_ref(StorageRef)
    return storage


def reduce_storage(storage):
    from . import get_sharing_strategy
    if storage.is_cuda:
        metadata = storage._share_cuda_()
        cache_key = metadata[1]
        rebuild = rebuild_storage_cuda
    elif get_sharing_strategy() == 'file_system':
        metadata = storage._share_filename_()
        cache_key = metadata[1]
        rebuild = rebuild_storage_filename
        storage._shared_incref()
    else:
        fd, size = storage._share_fd_()
        if sys.version_info[0] == 2:
            df = multiprocessing.reduction.reduce_handle(fd)
        else:
            df = multiprocessing.reduction.DupFd(fd)
        cache_key = fd_id(fd)
        metadata = (df, size)
        rebuild = rebuild_storage_fd

    shared_cache[cache_key] = storage._weak_ref(StorageRef)
    return (rebuild, (type(storage),) + metadata)


def init_reductions():
    ForkingPickler.register(torch.cuda.Event, reduce_event)

    for t in torch._storage_classes:
        ForkingPickler.register(t, reduce_storage)

    for t in torch._tensor_classes:
        ForkingPickler.register(t, reduce_tensor)
