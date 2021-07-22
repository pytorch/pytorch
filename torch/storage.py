import io

import torch
from ._utils import _type, _cuda
from typing import Any, TypeVar, Type

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

    def type(self):
        return _type(self)

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


def _load_from_bytes(b):
    return torch.load(io.BytesIO(b))


_StorageBase.cuda = _cuda  # type: ignore[assignment]


def _dtype_to_pickle_storage_type_map():
    return {
        torch.double: 'DoubleStorage',
        torch.float: 'FloatStorage',
        torch.half: 'HalfStorage',
        torch.long: 'LongStorage',
        torch.int: 'IntStorage',
        torch.short: 'ShortStorage',
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

class TypedStorage:
    def __init__(self, storage, dtype: torch.dtype):

        if not isinstance(storage, torch.ByteStorage) and not isinstance(storage, torch.cuda.ByteStorage):
            raise RuntimeError(
                f'expected a ByteStorage, but got {type(storage)}')

        self.storage = storage
        self.dtype = dtype

    def __str__(self):
        return f'TypedStorage(storage={self.storage}, dtype={self.dtype})'

    def __iter__(self):
        return self.storage.__iter__()

    def __len__(self):
        return self.storage.__len__()

    def numel(self):
        element_size = torch._utils._element_size(self.dtype)
        return self.storage.nbytes() // element_size

    def nbytes(self):
        return self.storage.nbytes()

    def pickle_storage_type(self):
        try:
            return _dtype_to_pickle_storage_type_map()[self.dtype]
        except KeyError:
            raise KeyError(f'dtype {self.dtype} is not recognized')

    def __reduce__(self):
        b = io.BytesIO()
        torch.save(self, b, _use_new_zipfile_serialization=False)
        return (_load_from_bytes, (b.getvalue(),))


def _get_dtype_from_pickle_storage_type(pickle_storage_type: str):
    try:
        return _pickle_storage_type_to_dtype_map()[pickle_storage_type]
    except KeyError:
        raise KeyError(
            f'pickle storage type "{pickle_storage_type}" is not recognized')
