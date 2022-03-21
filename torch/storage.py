import io

import torch
from ._utils import _type, _cuda
from torch.types import Storage
from typing import Any, TypeVar, Type, Union, cast
import copy
import collections
from functools import lru_cache

T = TypeVar('T', bound='Union[_StorageBase, _TypedStorage]')
class _StorageBase(object):
    _cdata: Any
    is_cuda: bool = False
    is_sparse: bool = False
    is_sparse_csr: bool = False
    device: torch.device

    def __init__(self, *args, **kwargs): ...  # noqa: E704
    def __len__(self) -> int: ...  # noqa: E704
    def __getitem__(self, idx): ...  # noqa: E704
    def copy_(self, source: T) -> T: ...  # noqa: E704
    def nbytes(self) -> int: ...  # noqa: E704

    def size(self) -> int:
        return self.nbytes()

    def type(self, dtype: str = None, non_blocking: bool = False) -> T: ...  # noqa: E704
    def cuda(self, device=None, non_blocking=False, **kwargs) -> T: ...  # noqa: E704
    def element_size(self) -> int: ...  # noqa: E704
    def get_device(self) -> int: ...  # noqa: E704
    def data_ptr(self) -> int: ...  # noqa: E704

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
        return iter(map(lambda i: self[i], range(self.size())))

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
        return super(_StorageBase, self).__sizeof__() + self.size()

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

    def _to(self, dtype):
        storage = torch.tensor([], dtype=torch.uint8, device=self.device).set_(cast(Storage, self)).to(dtype).storage()
        if storage.data_ptr() == self.data_ptr():
            storage = storage.clone()
        return storage

    def double(self):
        """Casts this storage to double type"""
        return self._to(torch.double)

    def float(self):
        """Casts this storage to float type"""
        return self._to(torch.float)

    def half(self):
        """Casts this storage to half type"""
        return self._to(torch.half)

    def long(self):
        """Casts this storage to long type"""
        return self._to(torch.long)

    def int(self):
        """Casts this storage to int type"""
        return self._to(torch.int)

    def short(self):
        """Casts this storage to short type"""
        return self._to(torch.short)

    def char(self):
        """Casts this storage to char type"""
        return self._to(torch.int8)

    def byte(self):
        """Casts this storage to byte type"""
        return self._to(torch.uint8)

    def bool(self):
        """Casts this storage to bool type"""
        return self._to(torch.bool)

    def bfloat16(self):
        """Casts this storage to bfloat16 type"""
        return self._to(torch.bfloat16)

    def complex_double(self):
        """Casts this storage to complex double type"""
        return self._to(torch.cdouble)

    def complex_float(self):
        """Casts this storage to complex float type"""
        return self._to(torch.cfloat)

    def pin_memory(self):
        """Copies the storage to pinned memory, if it's not already pinned."""
        if self.is_cuda:
            raise TypeError(f"cannot pin '{self.type()}' only CPU memory can be pinned")
        import torch.cuda
        allocator = torch.cuda._host_allocator()  # type: ignore[attr-defined]
        return type(self)(self.size(), allocator=allocator).copy_(self)

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
    def _new_shared(cls, size):
        """Creates a new storage in shared memory with the same data type"""
        from torch.multiprocessing import get_sharing_strategy
        if cls.is_cuda:
            return cls(size)
        elif get_sharing_strategy() == 'file_system':
            return cls._new_using_filename(size)
        else:
            return cls._new_using_fd(size)

    def _untyped(self):
        return self


def _load_from_bytes(b):
    return torch.load(io.BytesIO(b))


_StorageBase.type = _type  # type: ignore[assignment]
_StorageBase.cuda = _cuda  # type: ignore[assignment]


@lru_cache(maxsize=None)
def _dtype_to_storage_type_map():
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
        torch.quint2x4: 'QUInt2x4Storage',
    }

@lru_cache(maxsize=None)
def _storage_type_to_dtype_map():
    dtype_map = {
        val: key for key, val in _dtype_to_storage_type_map().items()}
    return dtype_map

class _TypedStorage:
    is_sparse = False

    def fill_(self, value):
        self[0:len(self)] = value
        return self

    def __init__(self, *args, **kwargs):
        arg_error_msg = (
            f'{type(self)} constructor received an invalid combination '
            f'of arguments - got args={tuple(type(arg) for arg in args)}, '
            f'kwargs={ {key: type(val) for key, val in kwargs.items()} }, but '
            'expected one of:\n'
            ' * no arguments\n'
            ' * (int size)\n'
            ' * (Sequence data)\n')
        if type(self) == _TypedStorage:
            arg_error_msg += ' * (wrap_storage=<_UntypedStorage>, dtype=<torch.dtype>)'
        else:
            arg_error_msg += ' * (wrap_storage=<_UntypedStorage>)'

        if 'wrap_storage' in kwargs:
            assert len(args) == 0, (
                "No positional arguments should be given when using "
                "'wrap_storage'")

            if type(self) == _TypedStorage:
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

            if not isinstance(storage, (torch._UntypedStorage, torch.cuda._UntypedStorage)):
                raise TypeError(arg_error_msg)
            if type(self) != _TypedStorage and storage.__module__ != self.__module__:
                raise TypeError((
                    arg_error_msg +
                    f'\n`storage` `module {storage.__module__}` does not match '
                    f'module of {type(self)}'))
            self._storage = storage

        else:
            assert type(self) != _TypedStorage, (
                "Calling __init__ this way is only supported in _TypedStorage's "
                "child classes. _TypedStorage can only be directly instantiated "
                "when kwargs 'wrap_storage' and 'dtype' are given.")

            assert len(kwargs) == 0, "invalid keyword arguments"

            def isint(x):
                try:
                    int(x)
                except TypeError:
                    return False
                return True

            if len(args) == 0:
                self._storage = eval(self.__module__)._UntypedStorage()

            elif len(args) == 1 and isint(args[0]):
                self._storage = eval(self.__module__)._UntypedStorage(int(args[0]) * self.element_size())

            elif len(args) == 1 and isinstance(args[0], collections.abc.Sequence):
                if self.dtype in [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]:
                    interpret_dtypes = {
                        torch.quint8: torch.uint8,
                        torch.quint4x2: torch.uint8,
                        torch.quint2x4: torch.uint8,
                        torch.qint32: torch.int32,
                        torch.qint8: torch.int8
                    }
                    tmp_tensor = torch.tensor(
                        args[0],
                        dtype=interpret_dtypes[self.dtype],
                        device='cuda' if eval(self.__module__) is torch.cuda else 'cpu')

                else:
                    tmp_tensor = torch.tensor(
                        args[0],
                        dtype=self.dtype,
                        device='cuda' if eval(self.__module__) is torch.cuda else 'cpu')

                self._storage = tmp_tensor.storage()._untyped()

            else:
                raise TypeError(arg_error_msg)

    @property
    def is_cuda(self):
        return self._storage.device.type == 'cuda'

    def _untyped(self):
        return self._storage

    def _new_wrapped_storage(self, untyped_storage):
        module = eval(untyped_storage.__module__)
        assert type(untyped_storage) == module._UntypedStorage

        if type(self) == _TypedStorage:
            return _TypedStorage(wrap_storage=untyped_storage, dtype=self.dtype)
        else:
            # NOTE: We need to use the module of untyped_storage in case self's
            # module is different, e.g. if self is on CPU and untyped_storage
            # is on CUDA, and vice versa
            return getattr(module, type(self).__name__)(wrap_storage=untyped_storage)

    def __len__(self):
        return self._storage.nbytes() // self.element_size()

    def _maybe_wrap_index(self, idx, is_stop=False):
        if idx is None:
            if is_stop:
                return self.size()
            else:
                return 0

        else:
            if type(idx) != int:
                raise TypeError(
                    f"can't index a {type(self)} with {type(idx)}")
            if is_stop:
                if (idx > self.size()) or (idx < -self.size()):
                    raise IndexError(
                        f'index {idx} out of range for storage of size {self.size()}')
                if idx > 0:
                    return idx
                else:
                    return idx % self.size()
            else:
                if (idx >= self.size()) or (idx < -self.size()):
                    raise IndexError(
                        f'index {idx} out of range for storage of size {self.size()}')
                return idx % self.size()

    def __setitem__(self, idx, value):
        if not isinstance(idx, (int, slice)):
            raise RuntimeError(f"can't index a {type(self)} with {type(idx)}")
        if self.dtype in [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]:
            interpret_dtypes = {
                torch.quint8: torch.uint8,
                torch.quint4x2: torch.uint8,
                torch.quint2x4: torch.uint8,
                torch.qint32: torch.int32,
                torch.qint8: torch.int8
            }
            tmp_dtype = interpret_dtypes[self.dtype]
            tmp_tensor = torch.tensor([], dtype=tmp_dtype, device=self.device).set_(_TypedStorage(
                wrap_storage=self._storage,
                dtype=tmp_dtype))
        else:
            tmp_tensor = torch.tensor([], dtype=self.dtype, device=self.device).set_(self)

        tmp_tensor[idx] = value

    def __getitem__(self, idx):
        # NOTE: Before _TypedStorage existed, indexing with a slice used to be
        # possible for <type>Storage objects. However, it would return
        # a storage view, which would be a hassle to implement in _TypedStorage,
        # so it was disabled
        if isinstance(idx, slice):
            raise RuntimeError('slices are only supported in _UntypedStorage.__getitem__')
        elif not isinstance(idx, int):
            raise RuntimeError(f"can't index a {type(self)} with {type(idx)}")

        if self.dtype in [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]:
            interpret_dtypes = {
                torch.quint8: torch.uint8,
                torch.quint4x2: torch.uint8,
                torch.quint2x4: torch.uint8,
                torch.qint32: torch.int32,
                torch.qint8: torch.int8
            }
            return _TypedStorage(
                wrap_storage=self._storage,
                dtype=interpret_dtypes[self.dtype])[idx]

        idx_wrapped = self._maybe_wrap_index(idx)
        tmp_tensor = torch.tensor([], dtype=self.dtype, device=self.device).set_(self)
        return tmp_tensor[idx_wrapped].item()

    def copy_(self, source: T, non_blocking=None):
        self._storage.copy_(source._untyped(), non_blocking)
        return self

    def nbytes(self):
        return self._storage.nbytes()

    def type(self, dtype: str = None, non_blocking: bool = False) -> Union[T, str]:
        if dtype is None:
            return '.'.join([self.__module__, type(self).__name__])
        else:
            return self._storage.type(dtype, non_blocking)

    def cuda(self, device=None, non_blocking=False, **kwargs) -> T:
        cuda_storage = self._storage.cuda(device, non_blocking, **kwargs)
        return self._new_wrapped_storage(cuda_storage)

    def element_size(self):
        return torch._utils._element_size(self.dtype)

    def get_device(self) -> int:
        return self._storage.get_device()

    def __str__(self):
        data_str = ' ' + '\n '.join(str(self[i]) for i in range(self.size()))
        if type(self) == _TypedStorage:
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
        return self._new_wrapped_storage(copy.copy(self._storage))

    def __deepcopy__(self, memo):
        return self._new_wrapped_storage(copy.deepcopy(self._storage, memo))

    def __sizeof__(self):
        return super(_TypedStorage, self).__sizeof__() + self.nbytes()

    def clone(self):
        """Returns a copy of this storage"""
        return self._new_wrapped_storage(self._storage.clone())

    def tolist(self):
        """Returns a list containing the elements of this storage"""
        return list(self)

    def cpu(self):
        """Returns a CPU copy of this storage if it's not already on the CPU"""
        return self._new_wrapped_storage(self._storage.cpu())

    def pin_memory(self):
        """Coppies the  storage to pinned memory, if it's not already pinned."""
        return self._new_wrapped_storage(self._storage.pin_memory())

    def share_memory_(self):
        """Moves the storage to shared memory.

        This is a no-op for storages already in shared memory and for CUDA
        storages, which do not need to be moved for sharing across processes.
        Storages in shared memory cannot be resized.

        Returns: self
        """
        self._storage.share_memory_()
        return self

    @classmethod
    def _new_shared(cls, size):
        """Creates a new storage in shared memory with the same data type"""
        module = eval(cls.__module__)
        untyped_storage = module._UntypedStorage._new_shared(size * cls().element_size())
        return cls(wrap_storage=untyped_storage)

    @property
    def _cdata(self):
        return self._storage._cdata

    @property
    def device(self):
        return self._storage.device

    def size(self):
        return len(self)

    def pickle_storage_type(self):
        try:
            return _dtype_to_storage_type_map()[self.dtype]
        except KeyError:
            raise KeyError(f'dtype {self.dtype} is not recognized')

    def __reduce__(self):
        b = io.BytesIO()
        torch.save(self, b, _use_new_zipfile_serialization=False)
        return (_load_from_bytes, (b.getvalue(),))

    def data_ptr(self):
        return self._storage.data_ptr()

    def resize_(self, size):
        self._storage.resize_(size * self.element_size())

    @classmethod
    def _free_weak_ref(cls, *args, **kwargs):
        return eval(cls.__module__)._UntypedStorage._free_weak_ref(*args, **kwargs)

    def _weak_ref(self, *args, **kwargs):
        return self._storage._weak_ref(*args, **kwargs)

    @classmethod
    def from_buffer(cls, *args, **kwargs):
        if cls == _TypedStorage:
            raise RuntimeError(
                'from_buffer: only supported for subclasses of _TypedStorage')

        if 'dtype' in kwargs or len(args) == 5:
            raise RuntimeError((
                "from_buffer: 'dtype' can only be specified in "
                "_UntypedStorage.from_buffer"))

        kwargs['dtype'] = cls().dtype

        untyped_storage = eval(cls.__module__)._UntypedStorage.from_buffer(*args, **kwargs)
        return cls(wrap_storage=untyped_storage)

    def _to(self, dtype):
        storage = torch.tensor([], dtype=self.dtype, device=self.device).set_(self).to(dtype).storage()
        if storage.data_ptr() == self.data_ptr():
            storage = storage.clone()
        return storage

    def double(self):
        """Casts this storage to double type"""
        return self._to(torch.double)

    def float(self):
        """Casts this storage to float type"""
        return self._to(torch.float)

    def half(self):
        """Casts this storage to half type"""
        return self._to(torch.half)

    def long(self):
        """Casts this storage to long type"""
        return self._to(torch.long)

    def int(self):
        """Casts this storage to int type"""
        return self._to(torch.int)

    def short(self):
        """Casts this storage to short type"""
        return self._to(torch.short)

    def char(self):
        """Casts this storage to char type"""
        return self._to(torch.int8)

    def byte(self):
        """Casts this storage to byte type"""
        return self._to(torch.uint8)

    def bool(self):
        """Casts this storage to bool type"""
        return self._to(torch.bool)

    def bfloat16(self):
        """Casts this storage to bfloat16 type"""
        return self._to(torch.bfloat16)

    def complex_double(self):
        """Casts this storage to complex double type"""
        return self._to(torch.cdouble)

    def complex_float(self):
        """Casts this storage to complex float type"""
        return self._to(torch.cfloat)

    @classmethod
    def from_file(cls, filename, shared, size):
        if cls == _TypedStorage:
            raise RuntimeError('from_file can only be called on derived classes')
        untyped_storage = eval(cls.__module__)._UntypedStorage.from_file(
            filename,
            shared,
            size * torch._utils._element_size(cls.dtype))
        storage = cls(wrap_storage=untyped_storage)
        return storage

    @classmethod
    def _expired(cls, *args, **kwargs):
        return eval(cls.__module__)._UntypedStorage._expired(*args, **kwargs)

    def is_pinned(self):
        return self._storage.is_pinned()

    def _write_file(self, *args, **kwargs):
        return self._storage._write_file(*args, **kwargs)

    def _set_from_file(self, *args, **kwargs):
        return self._storage._set_from_file(*args, **kwargs)

    def _set_cdata(self, *args, **kwargs):
        return self._storage._set_cdata(*args, **kwargs)

    def _share_cuda_(self, *args, **kwargs):
        return self._storage._share_cuda_(*args, **kwargs)

    def is_shared(self):
        return self._storage.is_shared()

    @classmethod
    def _new_shared_cuda(cls, *args, **kwargs):
        return eval(cls.__module__)._UntypedStorage._new_shared_cuda(*args, **kwargs)

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs):
        return eval(cls.__module__)._UntypedStorage._new_with_weak_ptr(*args, **kwargs)

    def _share_filename_(self, *args, **kwargs):
        manager_handle, storage_handle, size = self._storage._share_filename_(*args, **kwargs)
        return manager_handle, storage_handle, size // self.element_size()

    @classmethod
    def _new_shared_filename(cls, manager, obj, size):
        bytes_size = size * torch._utils._element_size(cls.dtype)
        return cls(wrap_storage=eval(cls.__module__)._UntypedStorage._new_shared_filename(manager, obj, bytes_size))

    def _shared_decref(self):
        self._storage._shared_decref()
        return self

    @classmethod
    def _release_ipc_counter(cls, *args, **kwargs):
        return eval(cls.__module__)._UntypedStorage._release_ipc_counter(*args, **kwargs)

    def _shared_incref(self, *args, **kwargs):
        return self._storage._shared_incref(*args, **kwargs)

    def _share_fd_(self, *args, **kwargs):
        fd, size = self._storage._share_fd_(*args, **kwargs)
        return fd, size // self.element_size()

def _get_dtype_from_pickle_storage_type(pickle_storage_type: str):
    try:
        return _storage_type_to_dtype_map()[pickle_storage_type]
    except KeyError:
        raise KeyError(
            f'pickle storage type "{pickle_storage_type}" is not recognized')
