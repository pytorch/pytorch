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


# mapping from handles to StorageRef objects. Doesn't include CUDA handles.
shared_cache = weakref.WeakValueDictionary()


# CUDA storage caching strategy
# -----------------------------
#
# All CUDA storages are cached into 'cuda_cache'.
#
# Call a "CUDA storage that processes wish to share" a "real storage".
# "real storages" are the storages that other processes want to and
# eventually will receive.
#
# Call a "CUDA storage that points to the base of an allocation block" a
# "base storage". These are obtained by opening CUDA handles to the
# base of an allocation block, where multiple "real storages" may live.
# Those "real storages" can be obtained by adding an offset to "base storage"
#
# "base storages" are cached into the "base cache" and "real storages" are
# cached into the "real cache"
#
# When a process shares a cuda storage:
# - It looks in the "real cache" for the storage.
# - If the storage is not found, we go into cpp-land to share the storage.
# - If the storage is found, we return the "real storage"'s metadata
#   (device, handle, size, offset, view_size) that was previously obtained
#   when cpp-land shared the storage.
#
# When a process receives a cuda storage:
# - It looks in the "real cache" for the real storage.
# - If it's not there, look in the "base cache".
# - If it exists in the "base cache", recreate the real storage, add it to the
#   "real cache", and return it.
# - If it doesn't exist in the "base cache", recreate the real and base storages
#   in cpp-land and add them to their respective caches.
#
# "base storages" are uniquely identified by their CUDA handle.
# "real storages" are uniquely identified by (handle, offset).
class CUDASharedCache:
    def __init__(self):
        # (handle) -> "base storage" ref
        self.base_cache = weakref.WeakValueDictionary()

        # (handle, offset) -> "real storage" ref
        self.real_cache = weakref.WeakValueDictionary()

        # dataptr -> "real storage" ref
        self.data_cache = weakref.WeakValueDictionary()

    @staticmethod
    def real_key(handle, offset):
        return '{}_{}'.format(handle, offset)

    def save_base(self, base_storage, handle):
        storageref = base_storage._weak_ref(StorageRef)
        self.base_cache[handle] = storageref

    def save_real(self, real_storage, metadata):
        _, handle, _, offset, _ = metadata

        # Saving metadata so this process does not need to re-get
        # the IPC handle in cpp-land when re-sharing this storage.
        storageref = real_storage._weak_ref(StorageRef)
        storageref.__real_metadata = metadata

        key = self.real_key(handle, offset)
        dataptr = real_storage.data_ptr()

        self.real_cache[key] = storageref
        self.data_cache[dataptr] = storageref

    def get_storage(self, cls, metadata):
        _, handle, _, offset, view_size = metadata

        # Case 1: "real storage" cached
        key = self.real_key(handle, offset)
        real_ref = self.real_cache.get(key)
        if real_ref is not None:
            return cls._new_with_weak_ptr(real_ref)

        # Case 3: Neither "real storage" nor "base storage" cached
        base_ref = self.base_cache.get(key)
        if base_ref is None:
            return None

        # Case 2: "base storage" cached. Recreate "real storage"
        base_storage = cls._new_with_weak_ptr(base_ref)
        real_storage = base_storage._new_view(offset, view_size)
        self.save_real(real_storage, metadata)
        return real_storage

    def get_metadata(self, storage):
        realref = self.data_cache.get(storage.data_ptr())
        if realref is None:
            return None
        return realref.__real_metadata


cuda_shared_cache = CUDASharedCache()


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


def save_real_storage(storage, handle, offset):
    real_cuda_storage_cache.save(storage, handle, offset)


def real_cuda_storage_from_cache(cls, handle, offset):
    real_storage_ref = real_cuda_storage_cache.get(handle, offset)
    if real_storage_ref is None:
        return None
    return cls._new_with_weak_ptr(real_storage_ref)


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
    metadata = (device, handle, size, offset, view_size)
    real_storage = cuda_shared_cache.get_storage(cls, metadata)
    if real_storage is not None:
        return real_storage

    torch.cuda._lazy_init()
    real_storage = cls._new_shared_cuda(device, handle, size, offset, view_size)
    base_storage, _ = real_storage._root_storage()
    cuda_shared_cache.save_real(real_storage, metadata)
    cuda_shared_cache.save_base(base_storage, handle)
    return real_storage


def rebuild_storage_empty(cls):
    return cls()


def share_cuda(storage):
    # If we've already shared this storage, no need to share again.
    metadata = cuda_shared_cache.get_metadata(storage)
    if metadata is not None:
        return metadata

    # The storage hasn't been shared before.
    metadata = storage._share_cuda_()
    cuda_shared_cache.save_real(storage, metadata)
    return metadata


def reduce_storage(storage):
    from . import get_sharing_strategy
    if storage.is_cuda:
        metadata = share_cuda(storage)
        return (rebuild_storage_cuda, (type(storage),) + metadata)
    elif get_sharing_strategy() == 'file_system':
        metadata = storage._share_filename_()
        cache_key = metadata[1]
        rebuild = rebuild_storage_filename
        storage._shared_incref()
    elif storage.size() == 0:
        # This is special cased because Empty tensors
        # (with size 0) cannot be mmapped.
        return (rebuild_storage_empty, (type(storage),))
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
