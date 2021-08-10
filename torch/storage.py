import io

import torch
from ._utils import _type, _cuda
from typing import Any, TypeVar, Type
import copy

T = TypeVar('T', bound='_StorageBase')
class _StorageBase(object):
    _cdata: Any
    is_cuda: bool = False
    is_sparse: bool = False

    def __init__(self, *args, **kwargs): ...  # noqa: E704
    def __len__(self) -> int: ...  # noqa: E704
    def __getitem__(self, idx): ...  # noqa: E704
    def copy_(self, source: T) -> T: ...  # noqa: E704
    def nbytes(self) -> int: ...  # noqa: E704
    def type(self, dtype: str = None, non_blocking: bool = False) -> T: ...  # noqa: E704
    def cuda(self, device=None, non_blocking=False, **kwargs) -> T: ...  # noqa: E704
    def element_size(self) -> int: ...  # noqa: E704
    def get_device(self) -> int: ...  # noqa: E704

    # Defined in torch/csrc/generic/StorageSharing.cpp
    def _share_filename_(self): ...  # noqa: E704
    def _share_fd_(self): ...  # noqa: E704
    @classmethod
    def _new_using_filename(cls: Type[T], size: int) -> T: ...  # noqa: E704
    @classmethod
    def _new_using_fd(cls: Type[T], size: int) -> T: ...  # noqa: E704

    def __str__(self):
        content = ' ' + '\n '.join(str(self[i]) for i in range(len(self)))
        return content + f'\n[{torch.typename(self)} of size {len(self)}]'

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(map(lambda i: self[i], range(self.nbytes())))

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memo):
        memo = memo.setdefault('torch', {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_storage = self.clone()
        memo[self._cdata] = new_storage
        return new_storage

    def __reduce__(self):
        b = io.BytesIO()
        torch.save(self, b, _use_new_zipfile_serialization=False)
        return (_load_from_bytes, (b.getvalue(),))

    def __sizeof__(self):
        return super(_StorageBase, self).__sizeof__() + self.nbytes()

    def clone(self):
        """Returns a copy of this storage"""
        device = self.get_device() if self.is_cuda else -1
        with torch.cuda.device(device):
            return type(self)(self.nbytes()).copy_(self)

    def tolist(self):
        """Returns a list containing the elements of this storage"""
        return list(self)

    def cpu(self):
        """Returns a CPU copy of this storage if it's not already on the CPU"""
        return _type(self, getattr(torch, self.__class__.__name__))

    def pin_memory(self):
        """Copies the storage to pinned memory, if it's not already pinned."""
        if self.is_cuda:
            raise TypeError(f"cannot pin '{self.type()}' only CPU memory can be pinned")
        import torch.cuda
        allocator = torch.cuda._host_allocator()  # type: ignore[attr-defined]
        return type(self)(self.nbytes(), allocator=allocator).copy_(self)

    def share_memory_(self):
        """Moves the storage to shared memory.

        This is a no-op for storages already in shared memory and for CUDA
        storages, which do not need to be moved for sharing across processes.
        Storages in shared memory cannot be resized.

        Returns: self
        """
        from torch.multiprocessing import get_sharing_strategy
        if self.is_cuda:
            pass  # CUDA doesn't use POSIX shared memory
        elif get_sharing_strategy() == 'file_system':
            self._share_filename_()
        else:
            self._share_fd_()
        return self

    @classmethod
    def _new_shared(cls, *, nbytes):
        """Creates a new storage in shared memory with the same data type"""
        from torch.multiprocessing import get_sharing_strategy
        if cls.is_cuda:
            return cls(nbytes)
        elif get_sharing_strategy() == 'file_system':
            return cls._new_using_filename(nbytes)
        else:
            return cls._new_using_fd(nbytes)

    def _untyped(self):
        return self

# This class defines methods that both TypedStorage and ByteStorage
# share
class _StorageOverrides():
    @classmethod
    def from_buffer(cls, *args, **kwargs):
        return _from_buffer_override(cls, *args, **kwargs)

    # TODO: Add the float(), double(), etc methods and make them work correctly

    #def float(self):
    #    return self.type(type(self).__module__ + '.FloatStorage')

    #def double(self):
    #    return self.type(type(self).__module__ + '.DoubleStorage')

def _from_buffer_override(cls, *args, **kwargs):
    storage = eval(cls.__module__ + '._ByteStorage').from_buffer(*args, **kwargs)
    storage.__class__ = cls
    return storage


def _load_from_bytes(b):
    return torch.load(io.BytesIO(b))


_StorageBase.type = _type  # type: ignore[assignment]
_StorageBase.cuda = _cuda  # type: ignore[assignment]


def _dtype_to_pickle_storage_type_map():
    return {
        torch.double: 'DoubleStorage',
        torch.float: 'FloatStorage',
        torch.half: 'HalfStorage',
        torch.long: 'LongStorage',
        torch.int: 'IntStorage',
        torch.int16: 'ShortStorage',
        torch.int8: 'CharStorage',
        torch.uint8: 'ByteStorage',
        torch.bool: 'BoolStorage',
        torch.bfloat16: 'BFloat16Storage',
        torch.cdouble: 'ComplexDoubleStorage',
        torch.cfloat: 'ComplexFloatStorage',
        torch.qint8: 'QInt8Storage',
        torch.qint32: 'QInt32Storage',
        torch.quint8: 'QUInt8Storage',
        torch.quint4x2: 'QUInt4x2Storage',
    }

def _pickle_storage_type_to_dtype_map():
    if not hasattr(_pickle_storage_type_to_dtype_map, 'cache'):
        _pickle_storage_type_to_dtype_map.cache = {  # type: ignore[attr-defined]
            val: key for key, val in _dtype_to_pickle_storage_type_map().items()}
    return _pickle_storage_type_to_dtype_map.cache  # type: ignore[attr-defined]

class TypedStorage(_StorageOverrides):
    is_sparse = False

    @property
    def is_cuda(self):
        return self.storage.is_cuda

    def fill_(self, value):
        self[0:len(self)] = value
        return self

    def __init__(self, *args, **kwargs):
        if 'wrap_storage' in kwargs:
            assert len(args) == 0, (
                "No positional arguments should be given when using "
                "'wrap_storage'")

            if type(self) == TypedStorage:
                assert 'dtype' in kwargs, (
                    "When using 'wrap_storage', 'dtype' also must be specified")
                assert len(kwargs) == 2, (
                    "Only 'wrap_storage' and 'dtype' should be given, but got: "
                    f"{kwargs}")
                dtype = kwargs['dtype']
                assert isinstance(dtype, torch.dtype)
                self.dtype = dtype

            else:
                assert hasattr(self, 'dtype')
                assert len(kwargs) == 1, (
                    f"Only 'wrap_storage' should be given, but got: {kwargs.keys()}")
                dtype = self.dtype

            storage = kwargs['wrap_storage']

            # TODO: change `self.storage` to `self._storage`
            if isinstance(storage, TypedStorage):
                # TODO: Consider whether we should actually allow this or not.
                # I think it might be useful for .float(), .double(), etc
                # methods?
                self.storage = storage.storage

            else:
                assert isinstance(storage, eval(storage.__module__ + ".ByteStorage"))
                self.storage = storage
        else:
            assert type(self) != TypedStorage, (
                "Calling __init__ this way is only supported in TypedStorage's "
                "child classes. TypedStorage can only be directly instantiated "
                "when kwargs 'wrap_storage' and 'dtype' are given.")

            # TODO: Still need to override some of the constructor behavior.
            # For instance, when we do something like
            # `torch.FloatStorage([0, 1, 2])`, we'll end up calling
            # `torch.ByteStorage([0, 1, 2])` here, which is not correct,
            # since we want the storage to be filled with 3 float values
            # [0, 1, 2]
            self.storage = eval(self.__module__ + ".ByteStorage")(*args, **kwargs)

    def _untyped(self):
        return self.storage

    def _new_wrapped_storage(self, byte_storage):
        module = eval(byte_storage.__module__)
        assert type(byte_storage) == module.ByteStorage

        if type(self) == TypedStorage:
            return TypedStorage(wrap_storage=byte_storage, dtype=self.dtype)
        else:
            # NOTE: We need to use the module of byte_storage in case self's
            # module is different. e.g. if self is on CPU and byte_storage
            # is on CUDA, and vice versa
            return getattr(module, type(self).__name__)(wrap_storage=byte_storage)

    def __len__(self):
        return self.storage.nbytes() // self.element_size()

    def _maybe_wrap_index(self, idx):
        if type(idx) != int:
            raise TypeError(
                f"can't index a {type(self)} with {type(idx)}")

        if idx is None:
            return 0

        else:
            if (idx >= self.size()) or (idx < -self.size()):
                raise IndexError(
                    f'index {idx} out of range for storage of size {self.size()}')
            return idx % self.size()


    def __setitem__(self, idx, value):
        # TODO: Seems bad to use a tensor inside a storage method
        tmp_tensor = torch.tensor(self.storage, device=self.device, dtype=self.dtype)
        tmp_tensor[idx] = value

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Need to wrap start and stop, then multiply each slice param
            # by element_size
            start = self.element_size() * self._maybe_wrap_index(idx.start)
            stop = self.element_size() * self._maybe_wrap_index(idx.stop)
            
            if idx.step is not None and idx.step != 1:
                raise RuntimeError((
                    f'Trying to slice with a step of {idx.step}, but '
                    'only a step of 1 is supported'))

            byte_slice = slice(start, stop, 1)
            byte_storage = self.storage[byte_slice]
            return self._new_wrapped_storage(byte_storage)

        else:
            idx_wrapped = self._maybe_wrap_index(idx)
            # TODO: Seems bad to use a tensor inside a storage method
            tmp_tensor = torch.tensor(self.storage, device=self.device, dtype=self.dtype)
            return tmp_tensor[idx_wrapped].item()

    def copy_(self, source: T, non_blocking=None) -> T:
        if isinstance(source, TypedStorage):
            source = source.storage

        self.storage.copy_(source, non_blocking)

        return self

    def nbytes(self):
        return self.storage.nbytes()

    def type(self, dtype: str = None, non_blocking: bool = False) -> T:
        return self.storage.type(dtype, non_blocking)

    def cuda(self, device=None, non_blocking=False, **kwargs) -> T:
        cuda_storage = self.storage.cuda(device, non_blocking, **kwargs)
        return self._new_wrapped_storage(cuda_storage)

    def element_size(self):
        return torch._utils._element_size(self.dtype)

    def get_device(self) -> int:
        return self.storage.get_device()

    #def _share_filename_(self):

    #def _share_fd_(self):

    #@classmethod
    #def _new_using_filename(cls: Type[T], size:int) -> T:

    #@classmethod
    #def _new_using_fd(cls: Type[T], size:int) -> T:

    def __str__(self):
        data_str = ' ' + '\n '.join(str(self[i]) for i in range(self.size()))
        if type(self) == TypedStorage:
            return data_str + (
                f'\n[{torch.typename(self)} with dtype {self.dtype} '
                f'of size {len(self)}]')
        else:
            return data_str + f'\n[{torch.typename(self)} of size {len(self)}]'

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(map(lambda i: self[i], range(self.size())))

    def __copy__(self):
        return self._new_wrapped_storage(copy.copy(self.storage))

    def __deepcopy__(self, memo):
        return self._new_wrapped_storage(copy.deepcopy(self.storage, memo))

    #def __reduce__(self):

    def __sizeof__(self):
        return super(TypedStorage, self).__sizeof__() + self.nbytes()

    def clone(self):
        """Returns a copy of this storage"""
        return self._new_wrapped_storage(self.storage.clone())

    def tolist(self):
        """Returns a list containing the elements of this storage"""
        return list(self)

    def cpu(self):
        """Returns a CPU copy of this storage if it's not already on the CPU"""
        return self._new_wrapped_storage(self.storage.cpu())

    def pin_memory(self):
        """Coppies the  storage to pinned memory, if it's not already pinned."""
        return self._new_wrapped_storage(self.storage.pin_memory())
    
    def share_memory_(self):
        """Moves the storage to shared memory.

        This is a no-op for storages already in shared memory and for CUDA
        storages, which do not need to be moved for sharing across processes.
        Storages in shared memory cannot be resized.

        Returns: self
        """
        self.storage.share_memory_()
        return self

    @classmethod
    def _new_shared(cls, *, nbytes):
        """Creates a new storage in shared memory with the same data type"""
        module = eval(cls.__module__)
        byte_storage = module.ByteStorage._new_shared(nbytes=nbytes)
        return cls(wrap_storage=byte_storage)

    @property
    def _cdata(self):
        self.storage._cdata

    @property
    def device(self):
        return self.storage.device

    def size(self):
        return len(self)

    def pickle_storage_type(self):
        try:
            return _dtype_to_pickle_storage_type_map()[self.dtype]
        except KeyError:
            raise KeyError(f'dtype {self.dtype} is not recognized')

    def __reduce__(self):
        b = io.BytesIO()
        torch.save(self, b, _use_new_zipfile_serialization=False)
        return (_load_from_bytes, (b.getvalue(),))

    def data_ptr(self):
        return self.storage.data_ptr()

    def resize_(self, size):
        self.storage.resize_(size * self.element_size())

    # is_pinned
    # _write_file
    # _set_from_file
    # from_buffer
    # from_file
    # get_device
    # _set_cdata


def _get_dtype_from_pickle_storage_type(pickle_storage_type: str):
    try:
        return _pickle_storage_type_to_dtype_map()[pickle_storage_type]
    except KeyError:
        raise KeyError(
            f'pickle storage type "{pickle_storage_type}" is not recognized')
