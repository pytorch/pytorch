import torch
import os
import weakref
import threading
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


class StorageWeakRef(object):
    r"""A weak reference to a Storage.

    The cdata member is a Python number containing the integer representation of
    the Storage pointer."""

    def __init__(self, storage):
        self.cdata = storage._weak_ref()
        # Save a direct reference to _free_weak_ref because the `torch` module
        # might be cleared during Python shutdown before this module is cleared.
        self._free_weak_ref = torch.Storage._free_weak_ref

    def expired(self):
        return torch.Storage._expired(self.cdata)

    def __del__(self):
        self._free_weak_ref(self.cdata)


class SharedCache(dict):
    """dictionary from multiprocessing handles to StorageWeakRef"""

    def __init__(self):
        # free_dead_references() is called if the len exceeds the currrent
        # limit. The limit scales with the number of remaining live objects.
        self.limit = 128
        self.lock = threading.Lock()

    def __setitem__(self, key, storage_ref):
        dict.__setitem__(self, key, storage_ref)
        if len(self) > self.limit:
            self.free_dead_references()

    def free_dead_references(self):
        # Multiple Python threads may call free_dead_references() concurrently.
        # Without a lock, they may try deleting the same entry multiple times.
        with self.lock:
            live = 0
            for key, storage_ref in list(self.items()):
                if storage_ref.expired():
                    del self[key]
                else:
                    live += 1
            self.limit = max(128, live * 2)


# mapping from handles to StorageWeakRef objects
shared_cache = SharedCache()


def rebuild_event(handle):
    return torch.cuda.Event(_handle=handle)


def reduce_event(event):
    return (rebuild_event, (event.ipc_handle(),))


def rebuild_tensor(cls, storage, metadata):
    storage_offset, size, stride, requires_grad, backward_hooks = metadata
    t = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
    if cls == torch.nn.parameter.Parameter:
        t = torch.nn.parameter.Parameter(t)
    t.requires_grad = requires_grad
    t._backward_hooks = backward_hooks
    return t


def rebuild_cuda_tensor(tensor_cls, tensor_size, tensor_stride, tensor_offset,
                        storage_cls, storage_device, storage_handle, storage_size, requires_grad, backward_hooks):

    storage = storage_from_cache(storage_cls, storage_handle)
    if storage is None:
        torch.cuda._lazy_init()
        storage = storage_cls._new_shared_cuda(storage_device, storage_handle, storage_size)
        shared_cache[storage_handle] = StorageWeakRef(storage)

    t = torch._utils._rebuild_tensor(storage, tensor_offset, tensor_size, tensor_stride)
    if tensor_cls == torch.nn.parameter.Parameter:
        t = torch.nn.parameter.Parameter(t)
    t.requires_grad = requires_grad
    t._backward_hooks = backward_hooks
    return t


def reduce_tensor(tensor):
    storage = tensor.storage()

    if tensor.requires_grad and not tensor.is_leaf:
        raise RuntimeError("Cowardly refusing to serialize non-leaf tensor which requires_grad, "
                           "since autograd does not support crossing process boundaries.  "
                           "If you just want to transfer the data, call detach() on the tensor "
                           "before serializing (e.g., putting it on the queue).")

    # Note [CUDA IPC and the caching allocator]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # When you send a CUDA tensor over IPC, you might expect that you will
    # get out the same storage from the other end.  However, the CUDA caching
    # allocator makes it difficult to preserve this invariant.  Consider
    # the following situation: a tensor of size 0x100 points to offset 0x20 of
    # a storage at 0xA100 of size 0x100.  (For simplicity, all of these
    # sizes are given in bytes).  HOWEVER, with the caching allocator, this storage
    # might be part of a larger cudaMalloc allocation 0xA000 of size 0x4000.
    #
    # When we want to send this CUDA tensor over IPC, we must send the
    # *entire* cudaMalloc allocation, i.e., the 0xA000 region, not just
    # the storage 0xA100 (because that is what CUDA supports).  So, on the
    # other end, there simply isn't any way to say, "Wait, you gave me
    # a bigger region (0xA000) than the one I wanted (0xA100)"; we have
    # to just make a storage for the entire caching allocator block.
    #
    # This is fine, because all we need to do is just adjust the offset
    # on the tensor itself: instead of:
    #
    #   Tensor(size=0x100, offset=0x020, storage=Storage(data=0xA100, size=0x0100))
    #
    # we have
    #
    #   Tensor(size=0x100, offset=0x120, storage=Storage(data=0xA000, size=0x4000))
    #
    # This strategy has a few implications:
    #
    # 1. When we serialize a CUDA tensor for IPC, we have to do it all in one
    #    go (non-compositionally), instead of first serializing storage, and
    #    then serializing tensor.  This is because the base address of the
    #    storage allocation affects what offset we write into the tensor.
    #
    # 2. We MUST NOT let the new IPC tensor be resizable.  Originally, a resize
    #    of the storage beyond 0x100 would merely have caused us to do a
    #    reallocation.  You don't really want to do this, but if you did,
    #    all that would happen is that you would lose IPC sharing.  But if
    #    you do this in the new world, we will happily let you write out of
    #    bounds of your "allocation", clobbering unrelated data in the cached
    #    allocator block.  BAD!
    #
    # By the way, in old versions of PyTorch, we supported this situation
    # natively using a "storage view", which permitted multiple storages to be
    # views on each other.  But this was the *only* use of storage views, so we
    # eliminated it so that we could just use tensor views to implement the same
    # thing.
    #
    if storage.is_cuda:
        (device, handle, storage_size, storage_offset) = storage._share_cuda_()
        tensor_offset = tensor.storage_offset()

        shared_cache[handle] = StorageWeakRef(storage)

        return (rebuild_cuda_tensor,
                (type(tensor),
                 tensor.size(),
                 tensor.stride(),
                 tensor_offset + storage_offset,
                 type(storage),
                 device,
                 handle,
                 storage_size,
                 tensor.requires_grad,
                 tensor._backward_hooks))

    metadata = (tensor.storage_offset(), tensor.size(), tensor.stride(), tensor.requires_grad, tensor._backward_hooks)
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
    return cls._new_with_weak_ptr(storage_ref.cdata)


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
        shared_cache[fd_id(fd)] = StorageWeakRef(storage)
        return storage
    finally:
        os.close(fd)


def rebuild_storage_filename(cls, manager, handle, size):
    storage = storage_from_cache(cls, handle)
    if storage is not None:
        return storage._shared_decref()
    storage = cls._new_shared_filename(manager, handle, size)
    shared_cache[handle] = StorageWeakRef(storage)
    return storage._shared_decref()


def rebuild_storage_empty(cls):
    return cls()


def reduce_storage(storage):
    from . import get_sharing_strategy
    if storage.is_cuda:
        raise RuntimeError("Cannot pickle CUDA storage; try pickling a CUDA tensor instead")
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

    shared_cache[cache_key] = StorageWeakRef(storage)
    return (rebuild, (type(storage),) + metadata)


def init_reductions():
    ForkingPickler.register(torch.cuda.Event, reduce_event)

    for t in torch._storage_classes:
        ForkingPickler.register(t, reduce_storage)

    for t in torch._tensor_classes:
        ForkingPickler.register(t, reduce_tensor)

    # TODO: Maybe this should be in tensor_classes? :)
    ForkingPickler.register(torch.Tensor, reduce_tensor)
    ForkingPickler.register(torch.nn.parameter.Parameter, reduce_tensor)
