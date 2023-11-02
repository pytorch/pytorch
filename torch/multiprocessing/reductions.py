import multiprocessing
import os
import threading
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Union

import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor

try:
    # Early load resource_sharer to prevent a partially initialized instance
    # from being inherited in a forked child process. The reduce_storage method
    # requires this module indirectly through DupFd(). The built-in mp.Queue
    # class pickles arguments in a background thread which may overlap with the
    # fork.
    import multiprocessing.resource_sharer
except ImportError:
    pass


class StorageWeakRef:
    r"""A weak reference to a Storage.

    The cdata member is a Python number containing the integer representation of
    the Storage pointer."""

    __slots__ = ["cdata", "_free_weak_ref"]

    def __init__(self, storage):
        self.cdata = storage._weak_ref()
        # Save a direct reference to _free_weak_ref because the `torch` module
        # might be cleared during Python shutdown before this module is cleared.
        self._free_weak_ref = torch.Storage._free_weak_ref  # type: ignore[attr-defined]

    @classmethod
    def from_weakref(cls, cdata):
        instance = cls.__new__(cls)
        instance.cdata = cdata
        instance._free_weak_ref = torch.Storage._free_weak_ref  # type: ignore[attr-defined]
        return instance

    def expired(self):
        return torch.Storage._expired(self.cdata)  # type: ignore[attr-defined]

    def __del__(self):
        self._free_weak_ref(self.cdata)

    def __hash__(self):
        return self.cdata

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        return self.cdata == other.cdata


class SharedCache(dict):
    """dictionary from multiprocessing handles to StorageWeakRef"""

    def __init__(self):
        # free_dead_references() is called if the len exceeds the current
        # limit. The limit scales with the number of remaining live objects.
        self.limit = 128
        # `fork` inherits lock state, so in case we fork when the lock is held,
        # we register a function to reset the lock to a new object to avoid
        # possible deadlocks, following python multiprocessing library design.
        self._after_fork()
        register_after_fork(self, SharedCache._after_fork)

    def _after_fork(self):
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            return dict.get(self, key)

    def __setitem__(self, key, storage_ref):
        with self.lock:
            dict.__setitem__(self, key, storage_ref)
            if len(self) > self.limit:
                self.free_dead_references()

    def free_dead_references(self):
        live = 0
        for key, storage_ref in list(self.items()):
            if storage_ref.expired():
                del self[key]
            else:
                live += 1
        self.limit = max(128, live * 2)


# mapping from handles to StorageWeakRef objects
shared_cache = SharedCache()


def rebuild_event(device, handle):
    return torch.cuda.Event.from_ipc_handle(device, handle)


def reduce_event(event):
    handle = event.ipc_handle()
    return (rebuild_event, (event.device, handle))


def rebuild_tensor(cls, storage, metadata):
    storage_offset, size, stride, requires_grad = metadata
    t = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
    if cls == torch.nn.parameter.Parameter:
        # we have to pass requires_grad into constructor, rather than set it as an
        # attribute later, because it's an important check for Integer Tensors to
        # have requires_grad=False (or else they raise an error)
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad
    return t


def rebuild_cuda_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    storage_cls,
    dtype,
    storage_device,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    requires_grad,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
):
    # If storage_handle is None, storage points to nullptr.
    if storage_handle is None or storage_size_bytes == 0:
        storage = storage_cls(0, dtype=dtype, device=storage_device, _internal=True)
    else:
        storage = storage_from_cache(
            storage_cls, (storage_handle, storage_offset_bytes)
        )
        if storage is None:
            torch.cuda._lazy_init()
            storage = storage_cls._new_shared_cuda(
                storage_device,
                storage_handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            )
            shared_cache[(storage_handle, storage_offset_bytes)] = StorageWeakRef(
                storage
            )
        else:
            # We already ref counting this Storage, but producer needs new ref-counters to be released.
            storage_cls._release_ipc_counter(
                ref_counter_handle, ref_counter_offset, device=storage_device
            )

    _storage = (
        storage
        if isinstance(storage, torch.UntypedStorage)
        else storage._untyped_storage
    )

    t = torch._utils._rebuild_tensor(
        torch.storage.TypedStorage(wrap_storage=_storage, dtype=dtype, _internal=True),
        tensor_offset,
        tensor_size,
        tensor_stride,
    )

    if tensor_cls == torch.nn.parameter.Parameter:
        # It is crucial for integer tensors to receive
        # the requires_grad=False as an argument in the constructor
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad

    return t


def reduce_tensor(tensor):
    storage = tensor._typed_storage()

    if tensor.requires_grad and not tensor.is_leaf:
        raise RuntimeError(
            "Cowardly refusing to serialize non-leaf tensor which requires_grad, "
            "since autograd does not support crossing process boundaries.  "
            "If you just want to transfer the data, call detach() on the tensor "
            "before serializing (e.g., putting it on the queue)."
        )

    check_serializing_named_tensor(tensor)
    torch.utils.hooks.warn_if_has_hooks(tensor)

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
    # a bigger region (0xA000) than the one I wanted (0xA100)".
    #
    # OK, so if you sent the cudaMalloc allocation, can you just wrap that up as
    # one storage itself? No, because this cudaMalloc allocation might contain
    # storages of mixed types: float, bytes, double... If you make the entire
    # allocation a single storage of a type A, we'll hit an error when constructing
    # a tensor of type B on the storage.
    #
    # cudaIpcMemHandle is an identifier to access the sender cudaMalloc allocation on the
    # receiver side. However, cudaIpcMemHandles from each device in a given process may
    # only be opened by one context per device per other process.
    # If we open and close a memory handle multiples times in a process, CUDA is allowed
    # to give it a different address; similarly, once we close the memory, we're not
    # allowed to access it(and the storage/tensor built on top of it), even if it is
    # still live in the original process. As we cannot make a cudaMalloc allocation
    # to a single storage in one go, this requires us to cache the device pointer for
    # each cudaIpcMemHandle on C++ side to reconstruct types of storages, while keep
    # the old ones alives.
    # See [https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html]
    #
    # This is fine, because all we need to do is to save our position in the allocation,
    # and reconstruct storage and tensor from it.
    # 0xA000 ->  -------CUDA Allocation------
    #           |                            |
    #           |                            |
    #           |                            |
    #           |                            |
    # 0xA100 ->  --------storage1 begin------
    #           |                            |
    # 0xA120 ->  --------tensor1 begin ------
    #           |                            |
    #           |                            |
    #           |                            |
    #           |                            |
    #           |                            |
    # 0xA160 ->  --------tensor1 end---------
    #           |                            |
    #           |                            |
    #           |                            |
    # 0xA200 ->  --------storage1 end--------
    #           |                            |
    # 0xE000 ->  --------CUDA allocation-----
    #
    # To send tensor1, the following info are required from sender to receiver for
    # storage recontruction.
    #   1. cudaIpcMemHandle of 0xA000(which can be mapped to a basePtr in receiver process).
    #      basePtr may not be exactly 0xA000 since it's a different process.
    #   2. offset(0xA100) of storage1 in the CUDA allocation.
    #   3. size of storage1(0x100).
    #
    # On receiver side:
    #   1. Get the devPtr of the MemHandle to access the memory, reconstruct a storage
    #      of the same type using (basePtr, offset, size).
    #   2. we can reconstruct the tensor on top of the reconstructed storage
    #   Tensor(size=0x040, offset=0x020, storage=Storage(data=basePtr+0xA100, size=0x0100))
    #
    # This strategy has a few implications:
    #
    # 1. When we serialize a CUDA tensor for IPC, we cannot do it all in one
    #    go (non-compositionally), and this requires to have a global map
    #    memHandle -> devPtr for each process.
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

    # TODO: Handle distinguishing between subclass and non-subclass versions of NT better
    # https://github.com/pytorch/pytorch/issues/110543
    from torch.nested._internal.nested_tensor import NestedTensor

    if tensor.is_nested and not isinstance(tensor, NestedTensor):
        return reduce_nested_tensor(tensor)

    if storage._untyped_storage.device.type == "cuda":
        (
            device,
            handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = storage._share_cuda_()
        tensor_offset = tensor.storage_offset()
        shared_cache[handle] = StorageWeakRef(storage)
        # _backward_hooks purposely omitted here, see
        # Note [Don't serialize hooks]
        return (
            rebuild_cuda_tensor,
            (
                type(tensor),
                tensor.size(),
                tensor.stride(),
                tensor_offset,  # tensor offset in its storage
                type(storage),
                tensor.dtype,
                device,
                handle,  # identifier which CUDA allocation is the storage in.
                storage_size_bytes,  # size(in bytes) of the storage
                storage_offset_bytes,  # offset(in bytes) of the storage in the CUDA allocation
                tensor.requires_grad,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ),
        )

    # _backward_hooks purposely omitted here, see Note [Don't serialize hooks]
    metadata = (
        tensor.storage_offset(),
        tensor.size(),
        tensor.stride(),
        tensor.requires_grad,
    )
    return (rebuild_tensor, (type(tensor), storage, metadata))


def rebuild_nested_tensor(
    rebuild_buffer_func,
    rebuild_buffer_args,
    rebuild_sizes_func,
    rebuild_sizes_args,
    rebuild_strides_func,
    rebuild_strides_args,
    rebuild_offsets_func,
    rebuild_offsets_args,
):
    buffer = rebuild_buffer_func(*rebuild_buffer_args)
    sizes = rebuild_sizes_func(*rebuild_sizes_args)
    strides = rebuild_strides_func(*rebuild_strides_args)
    offsets = rebuild_offsets_func(*rebuild_offsets_args)
    return torch._nested_view_from_buffer_copy(buffer, sizes, strides, offsets)


def reduce_nested_tensor(nt):
    rebuild_buffer_func, rebuild_buffer_args = reduce_tensor(nt.values())
    rebuild_sizes_func, rebuild_sizes_args = reduce_tensor(nt._nested_tensor_size())
    rebuild_strides_func, rebuild_strides_args = reduce_tensor(
        nt._nested_tensor_strides()
    )
    rebuild_offsets_func, rebuild_offsets_args = reduce_tensor(
        nt._nested_tensor_storage_offsets()
    )

    return (
        rebuild_nested_tensor,
        (
            rebuild_buffer_func,
            rebuild_buffer_args,
            rebuild_sizes_func,
            rebuild_sizes_args,
            rebuild_strides_func,
            rebuild_strides_args,
            rebuild_offsets_func,
            rebuild_offsets_args,
        ),
    )


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
    return torch.UntypedStorage._new_with_weak_ptr(storage_ref.cdata)


def rebuild_storage_fd(cls, df, size):
    fd = df.detach()
    try:
        storage = storage_from_cache(cls, fd_id(fd))
        if storage is not None:
            return storage
        storage = cls._new_shared_fd_cpu(fd, size)
        shared_cache[fd_id(fd)] = StorageWeakRef(storage)
        return storage
    finally:
        os.close(fd)


def rebuild_storage_filename(cls, manager, handle, size, dtype=None):
    storage: Union[torch.TypedStorage, torch.UntypedStorage] = storage_from_cache(
        cls, handle
    )
    if storage is not None:
        return storage._shared_decref()
    if dtype is None:
        storage = torch.UntypedStorage._new_shared_filename_cpu(manager, handle, size)
    else:
        byte_size = size * torch._utils._element_size(dtype)
        untyped_storage: torch.UntypedStorage = (
            torch.UntypedStorage._new_shared_filename_cpu(manager, handle, byte_size)
        )
        storage = torch.TypedStorage(
            wrap_storage=untyped_storage, dtype=dtype, _internal=True
        )
    shared_cache[handle] = StorageWeakRef(storage)
    return storage._shared_decref()


def rebuild_storage_empty(cls):
    return cls()


def rebuild_typed_storage(storage, dtype):
    return torch.storage.TypedStorage(wrap_storage=storage, dtype=dtype, _internal=True)


# Use for torch.storage.TypedStorage
def reduce_typed_storage(storage):
    return (rebuild_typed_storage, (storage._untyped_storage, storage.dtype))


def rebuild_typed_storage_child(storage, storage_type):
    return storage_type(wrap_storage=storage, _internal=True)


# Use for child classes of torch.storage.TypedStorage, like torch.FloatStorage
def reduce_typed_storage_child(storage):
    return (rebuild_typed_storage_child, (storage._untyped_storage, type(storage)))


def reduce_storage(storage):
    from . import get_sharing_strategy

    if storage.is_cuda:
        raise RuntimeError(
            "Cannot pickle CUDA storage; try pickling a CUDA tensor instead"
        )
    elif get_sharing_strategy() == "file_system":
        metadata = storage._share_filename_cpu_()
        cache_key = metadata[1]
        rebuild = rebuild_storage_filename
        if isinstance(storage, torch.TypedStorage):
            metadata += (storage.dtype,)
        storage._shared_incref()
    elif storage.size() == 0:
        # This is special cased because Empty tensors
        # (with size 0) cannot be mmapped.
        return (rebuild_storage_empty, (type(storage),))
    else:
        fd, size = storage._share_fd_cpu_()
        df = multiprocessing.reduction.DupFd(fd)
        cache_key = fd_id(fd)
        metadata = (df, size)
        rebuild = rebuild_storage_fd  # type: ignore[assignment]

    shared_cache[cache_key] = StorageWeakRef(storage)
    return (rebuild, (type(storage),) + metadata)


def init_reductions():
    ForkingPickler.register(torch.cuda.Event, reduce_event)

    for t in torch._storage_classes:
        if t.__name__ == "UntypedStorage":
            ForkingPickler.register(t, reduce_storage)
        else:
            ForkingPickler.register(t, reduce_typed_storage_child)

    ForkingPickler.register(torch.storage.TypedStorage, reduce_typed_storage)

    for t in torch._tensor_classes:
        ForkingPickler.register(t, reduce_tensor)

    # TODO: Maybe this should be in tensor_classes? :)
    ForkingPickler.register(torch.Tensor, reduce_tensor)
    ForkingPickler.register(torch.nn.parameter.Parameter, reduce_tensor)
