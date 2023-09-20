import io

import torch
from ._utils import _type, _cuda, _hpu
from torch.types import Storage
from typing import cast, Any, Dict as _Dict, Optional as _Optional, TypeVar, Type, Union
import copy
import collections
from functools import lru_cache
import warnings
import threading
import functools
try:
    import numpy as np
    HAS_NUMPY = True
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

_share_memory_lock = threading.Lock()
_share_memory_map: _Dict[int, threading.RLock] = {}

T = TypeVar('T', bound='Union[_StorageBase, TypedStorage]')
class _StorageBase:
    _cdata: Any
    is_sparse: bool = False
    is_sparse_csr: bool = False
    device: torch.device

    def __init__(self, *args, **kwargs): ...  # noqa: E704
    def __len__(self) -> int: ...  # type: ignore[empty-body] # noqa: E704
    def __getitem__(self, idx): ...  # noqa: E704
    def __setitem__(self, *args, **kwargs): ...  # noqa: E704
    def copy_(self, source: T, non_blocking: _Optional[bool] = None) -> T: ...  # type: ignore[empty-body] # noqa: E704
    def new(self) -> T: ...  # type: ignore[empty-body, misc, type-var] # noqa: E704
    def nbytes(self) -> int: ...  # type: ignore[empty-body] # noqa: E704

    def size(self) -> int:
        return self.nbytes()

    def type(self, dtype: _Optional[str] = None, non_blocking: bool = False) -> T: ...  # type: ignore[empty-body, misc, type-var] # noqa: E704
    def cuda(self, device=None, non_blocking=False, **kwargs) -> T: ...  # type: ignore[empty-body, misc, type-var] # noqa: E704
    def hpu(self, device=None, non_blocking=False, **kwargs) -> T: ...  # type: ignore[empty-body, misc, type-var] # noqa: E704
    def element_size(self) -> int: ...  # type: ignore[empty-body, type-var] # noqa: E704

    def get_device(self) -> int:
        return self.device.index

    def data_ptr(self) -> int: ...  # type: ignore[empty-body] # noqa: E704

    # Defined in torch/csrc/generic/StorageSharing.cpp
    def _share_filename_cpu_(self, *args, **kwargs): ...  # noqa: E704
    def _share_fd_cpu_(self, *args, **kwargs): ...  # noqa: E704
    @classmethod
    def _new_using_filename_cpu(cls: Type[T], size: int) -> T: ...  # type: ignore[empty-body] # noqa: E704
    @classmethod
    def _new_using_fd_cpu(cls: Type[T], size: int) -> T: ...  # type: ignore[empty-body] # noqa: E704
    @classmethod
    def from_buffer(cls: Type[T], *args, **kwargs) -> T: ...  # type: ignore[empty-body] # noqa: E704
    @classmethod
    def _new_shared_filename_cpu(cls: Type[T], manager, obj, size, *, device=None, dtype=None) -> T: ...  # type: ignore[empty-body] # noqa: E704
    @classmethod
    def _release_ipc_counter_cuda(cls: Type[T], *args, **kwargs) -> T: ...  # type: ignore[empty-body] # noqa: E704
    @classmethod
    def _new_with_weak_ptr(cls: Type[T], *args, **kwargs) -> T: ...  # type: ignore[empty-body] # noqa: E704
    def _shared_decref(self) -> T: ...  # type: ignore[empty-body, misc, type-var] # noqa: E704
    def _write_file(self, *args, **kwargs): ...  # noqa: E704
    def resize_(self, size: int): ...  # noqa: E704
    def _weak_ref(self, *args, **kwargs) -> T: ...  # type: ignore[empty-body, misc, type-var] # noqa: E704
    def _set_from_file(self, *args, **kwargs): ...  # noqa: E704
    def _set_cdata(self, *args, **kwargs): ...  # noqa: E704
    def _share_cuda_(self, *args, **kwargs): ...  # noqa: E704
    def is_shared(self) -> bool: ...  # type: ignore[empty-body] # noqa: E704
    @classmethod
    def _new_shared_cuda(cls: Type[T], *args, **kwargs) -> T: ...  # type: ignore[empty-body] # noqa: E704
    def _shared_incref(self, *args, **kwargs): ...  # noqa: E704
    @classmethod
    def _free_weak_ref(cls, *args, **kwargs): ...  # noqa: E704
    @property
    def is_cuda(self): ...  # noqa: E704
    @property
    def is_hpu(self): ...  # noqa: E704
    @classmethod
    def from_file(cls, filename, shared, nbytes) -> T: ...  # type: ignore[empty-body, misc, type-var] # noqa: E704
    @classmethod
    def _expired(cls, *args, **kwargs) -> T: ...  # type: ignore[empty-body, misc, type-var] # noqa: E704
    def _byteswap(self, *args, **kwargs): ...  # noqa: E704

    def __str__(self):
        info_str = (
            f'[{torch.typename(self)}(device={self.device}) '
            f'of size {len(self)}]')
        if self.device.type == 'meta':
            return '...\n' + info_str
        else:
            data_str = ' ' + '\n '.join(str(self[i]) for i in range(self.size()))
            return data_str + '\n' + info_str

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self[i] for i in range(self.size()))

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
        return super().__sizeof__() + self.size()

    def clone(self):
        """Returns a copy of this storage"""
        return type(self)(self.nbytes(), device=self.device).copy_(self)

    def tolist(self):
        """Returns a list containing the elements of this storage"""
        return list(self)

    def cpu(self):
        """Returns a CPU copy of this storage if it's not already on the CPU"""
        if self.device.type != 'cpu':
            return torch.UntypedStorage(self.size()).copy_(self, False)
        else:
            return self

    def mps(self):
        """Returns a MPS copy of this storage if it's not already on the MPS"""
        if self.device.type != 'mps':
            return torch.UntypedStorage(self.size(), device="mps").copy_(self, False)
        else:
            return self

    def _to(self, dtype):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Argument 'dtype' must be torch.dtype, not {type(dtype)}")
        storage = torch.tensor([], dtype=torch.uint8, device=self.device).set_(cast(Storage, self)).to(dtype)._typed_storage()
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

    def is_pinned(self, device: Union[str, torch.device] = 'cuda'):
        r"""Determine whether the CPU storage is already pinned on device.

        Args:
            device (str or torch.device): The device to pin memory on. Default: ``'cuda'``.

        Returns:
            A boolean variable.
        """
        return torch.tensor([], dtype=torch.uint8, device=self.device).set_(
            cast(Storage, self)).is_pinned(device)

    def pin_memory(self, device: Union[str, torch.device] = 'cuda'):
        r"""Copies the CPU storage to pinned memory, if it's not already pinned.

        Args:
            device (str or torch.device): The device to pin memory on. Default: ``'cuda'``.

        Returns:
            A pinned CPU storage.
        """
        if self.device.type != 'cpu':
            raise TypeError(f"cannot pin '{self.type()}' only CPU memory can be pinned")

        pinned_tensor = torch.tensor([], dtype=torch.uint8, device=self.device).set_(
            cast(Storage, self)).pin_memory(device)
        return pinned_tensor.untyped_storage()

    def share_memory_(self):
        """Moves the storage to shared memory.

        This is a no-op for storages already in shared memory and for CUDA
        storages, which do not need to be moved for sharing across processes.
        Storages in shared memory cannot be resized.

        Note that to mitigate issues like https://github.com/pytorch/pytorch/issues/95606
        it is thread safe to call this function from multiple threads on the same object.
        It is NOT thread safe though to call any other function on self without proper
        synchronization. Please see :doc:`/notes/multiprocessing` for more details.

        Returns: self
        """
        from torch.multiprocessing import get_sharing_strategy
        if self.device.type in ["cuda", torch._C._get_privateuse1_backend_name()]:
            pass  # CUDA or PrivateUse1 doesn't use POSIX shared memory
        elif get_sharing_strategy() == 'file_system':
            self._share_filename_cpu_()
        else:
            self._share_fd_cpu_()
        return self

    @classmethod
    def _new_shared(cls, size, *, device='cpu'):
        """Creates a new storage in shared memory with the same data type"""
        from torch.multiprocessing import get_sharing_strategy
        device = torch.device(device)
        if device.type in ["cuda", torch._C._get_privateuse1_backend_name()]:
            return cls(size, device=device)
        elif get_sharing_strategy() == 'file_system':
            return cls._new_using_filename_cpu(size)
        else:
            return cls._new_using_fd_cpu(size)

    def untyped(self):
        return self

    def byteswap(self, dtype):
        """Swaps bytes in underlying data"""
        elem_size = torch._utils._element_size(dtype)
        # for complex types, don't swap first and second numbers
        if dtype.is_complex:
            elem_size = max(int(elem_size / 2), 1)
        self._byteswap(elem_size)


def _share_memory_lock_protected(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        to_free = None
        to_wait = None
        with _share_memory_lock:
            key = self._cdata
            if key in _share_memory_map:
                to_wait = _share_memory_map[key]
            else:
                _share_memory_map[key] = threading.RLock()
                _share_memory_map[key].acquire()
                to_free = key

        # If we're already in the process of sharing the storage, wait
        # for it to be done.
        if to_wait is not None:
            with to_wait:
                pass

        try:
            return fn(self, *args, **kwargs)
        finally:
            # If we acquired the storage lock here and we're done working on it
            # we can now release it and free the entry.
            if to_free is not None:
                # Ensure that the cdata from the storage didn't change and only
                # the data_ptr did.
                assert self._cdata == to_free
                with _share_memory_lock:
                    _share_memory_map[to_free].release()
                    del _share_memory_map[to_free]
    return wrapper

class UntypedStorage(torch._C.StorageBase, _StorageBase):
    def __getitem__(self, *args, **kwargs):
        if self.device.type == 'meta':
            raise NotImplementedError("Not available for 'meta' device type")
        return super().__getitem__(*args, **kwargs)

    @property
    def is_cuda(self):
        return self.device.type == 'cuda'

    @property
    def is_hpu(self):
        return self.device.type == 'hpu'

    @_share_memory_lock_protected
    def share_memory_(self, *args, **kwargs):
        return super().share_memory_(*args, **kwargs)

    @_share_memory_lock_protected
    def _share_fd_cpu_(self, *args, **kwargs):
        return super()._share_fd_cpu_(*args, **kwargs)

    @_share_memory_lock_protected
    def _share_filename_cpu_(self, *args, **kwargs):
        return super()._share_filename_cpu_(*args, **kwargs)

def _load_from_bytes(b):
    return torch.load(io.BytesIO(b))


_StorageBase.type = _type  # type: ignore[assignment]
_StorageBase.cuda = _cuda  # type: ignore[assignment]
_StorageBase.hpu = _hpu  # type: ignore[assignment]


@lru_cache(maxsize=None)
def _dtype_to_storage_type_map():
    # NOTE: We should no longer add dtypes to this map. This map
    # is only used for BC/FC with older PyTorch versions. Going forward,
    # new dtypes of TypedStorage should not translate to a legacy
    # <type>Storage class. Instead, new dtypes of TypedStorage should
    # be serialized as an UntypedStorage paired with a torch.dtype
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

def _get_storage_from_sequence(sequence, dtype, device):
    if dtype in [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]:
        interpret_dtypes = {
            torch.quint8: torch.uint8,
            torch.quint4x2: torch.uint8,
            torch.quint2x4: torch.uint8,
            torch.qint32: torch.int32,
            torch.qint8: torch.int8
        }
        tmp_tensor = torch.tensor(
            sequence,
            dtype=interpret_dtypes[dtype],
            device=device)

    else:
        tmp_tensor = torch.tensor(
            sequence,
            dtype=dtype,
            device=device)

    return tmp_tensor._typed_storage()._untyped_storage

def _isint(x):
    if HAS_NUMPY:
        return isinstance(x, (int, np.integer))
    else:
        return isinstance(x, int)

_always_warn_typed_storage_removal = False

def _get_always_warn_typed_storage_removal():
    return _always_warn_typed_storage_removal

def _set_always_warn_typed_storage_removal(always_warn):
    global _always_warn_typed_storage_removal
    assert isinstance(always_warn, bool)
    _always_warn_typed_storage_removal = always_warn

def _warn_typed_storage_removal(stacklevel=2):
    global _always_warn_typed_storage_removal

    def is_first_time():
        if not hasattr(_warn_typed_storage_removal, 'has_warned'):
            return True
        else:
            return not _warn_typed_storage_removal.__dict__['has_warned']

    if _get_always_warn_typed_storage_removal() or is_first_time():
        message = (
            "TypedStorage is deprecated. It will be removed in the future and "
            "UntypedStorage will be the only storage class. This should only matter "
            "to you if you are using storages directly.  To access UntypedStorage "
            "directly, use tensor.untyped_storage() instead of tensor.storage()"
        )
        warnings.warn(message, UserWarning, stacklevel=stacklevel + 1)
        _warn_typed_storage_removal.__dict__['has_warned'] = True

def _reset_warn_typed_storage_removal():
    _warn_typed_storage_removal.__dict__['has_warned'] = False

def _get_device_from_module(module: str):
    if module.split(".")[-1] in ["cuda", torch._C._get_privateuse1_backend_name()]:
        return module.split(".")[-1]
    else:
        return "cpu"

class TypedStorage:
    is_sparse = False

    dtype: torch.dtype

    @property
    def _dtype(self):
        return self.dtype

    def fill_(self, value):
        _warn_typed_storage_removal()
        self._setitem(slice(0, self._size()), value)
        return self

    def __new__(cls, *args, wrap_storage=None, dtype=None, device=None, _internal=False):
        if not _internal:
            _warn_typed_storage_removal()

        if cls == torch.storage._LegacyStorage:
            raise RuntimeError("Only child classes of _LegacyStorage can be instantiated")

        if cls == TypedStorage:
            return super().__new__(cls)

        else:
            arg_error_msg = (
                f'{cls}.__new__ received an invalid combination '
                f'of arguments. Expected one of:\n'
                ' * no arguments\n'
                ' * (int size)\n'
                ' * (Sequence data)\n'
                ' * (*, UntypedStorage wrap_storage)')

            if device is not None:
                raise RuntimeError(
                    arg_error_msg +
                    "\nKeyword argument 'device' cannot be specified")

            if dtype is not None:
                raise RuntimeError(
                    arg_error_msg +
                    "\nKeyword argument 'dtype' cannot be specified")

            if wrap_storage is None:
                if len(args) > 1:
                    raise RuntimeError(
                        arg_error_msg +
                        "\nToo many positional arguments")

                if len(args) == 1 and not _isint(args[0]) and not isinstance(args[0], collections.abc.Sequence):
                    raise TypeError(
                        arg_error_msg +
                        f"\nArgument type not recognized: {type(args[0])}")

                return TypedStorage(
                    *args,
                    dtype=cls._dtype,
                    device=_get_device_from_module(cls.__module__),
                    _internal=True)

            else:
                if len(args) != 0:
                    raise RuntimeError(
                        arg_error_msg +
                        "\nNo positional arguments should be given when using "
                        "'wrap_storage'")

                if not isinstance(wrap_storage, torch.UntypedStorage):
                    raise TypeError(
                        arg_error_msg +
                        f"\nArgument 'wrap_storage' must be UntypedStorage, but got {type(wrap_storage)}")

                cls_device = _get_device_from_module(cls.__module__)

                if wrap_storage.device.type != cls_device:
                    raise RuntimeError(
                        arg_error_msg +
                        f"\nDevice of 'wrap_storage' must be {cls_device}"
                        f", but got {wrap_storage.device.type}")

                return TypedStorage(
                    *args,
                    wrap_storage=wrap_storage,
                    dtype=cls.dtype,
                    _internal=True)

    def __init__(self, *args, device=None, dtype=None, wrap_storage=None, _internal=False):
        if not _internal:
            _warn_typed_storage_removal()
        arg_error_msg = (
            'TypedStorage.__init__ received an invalid combination '
            'of arguments. Expected one of:\n'
            ' * (*, torch.device device, torch.dtype dtype)\n'
            ' * (int size, *, torch.device device, torch.dtype dtype)\n'
            ' * (Sequence data, *, torch.device device, torch.dtype dtype)\n'
            ' * (*, UntypedStorage wrap_storage, torch.dtype dtype)')

        if wrap_storage is not None:
            if len(args) != 0:
                raise RuntimeError(
                    arg_error_msg +
                    "\nNo positional arguments should be given when using "
                    "'wrap_storage'")

            if dtype is None:
                raise RuntimeError(
                    arg_error_msg +
                    "\nArgument 'dtype' must be specified")

            if not isinstance(dtype, torch.dtype):
                raise TypeError(
                    arg_error_msg +
                    f"\nArgument 'dtype' must be torch.dtype, not {type(dtype)}")

            if device is not None:
                raise RuntimeError(
                    arg_error_msg +
                    "\nArgument 'device' should not be specified when 'wrap_storage' is given")

            self.dtype = dtype

            if not isinstance(wrap_storage, torch.UntypedStorage):
                raise TypeError(
                    arg_error_msg +
                    f"\nArgument 'wrap_storage' must be UntypedStorage, but got {type(wrap_storage)}")

            self._untyped_storage = wrap_storage

        else:
            self.dtype = torch.get_default_dtype() if dtype is None else dtype
            device = torch.device('cpu' if device is None else device)

            if self.dtype in [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]:
                if device.type == 'cuda':
                    raise RuntimeError("Cannot create CUDA storage with quantized dtype")

            if len(args) == 0:
                self._untyped_storage = torch.UntypedStorage(device=device)

            elif len(args) == 1:
                if _isint(args[0]):
                    self._untyped_storage = torch.UntypedStorage(int(args[0]) * self._element_size(), device=device)
                elif isinstance(args[0], collections.abc.Sequence):
                    self._untyped_storage = _get_storage_from_sequence(args[0], self.dtype, device)
                else:
                    raise TypeError(
                        arg_error_msg +
                        f"\nArgument type not recognized: {type(args[0])}")

            else:
                raise RuntimeError(
                    arg_error_msg +
                    "\nToo many positional arguments")

    @property
    def is_cuda(self):
        _warn_typed_storage_removal()
        return self._untyped_storage.device.type == 'cuda'

    @property
    def is_hpu(self):
        _warn_typed_storage_removal()
        return self._untyped_storage.device.type == 'hpu'

    def untyped(self):
        """Returns the internal :class:`torch.UntypedStorage`"""
        _warn_typed_storage_removal()
        return self._untyped_storage

    def _new_wrapped_storage(self, untyped_storage):
        assert type(untyped_storage) == torch.UntypedStorage

        if type(self) == TypedStorage:
            return TypedStorage(
                wrap_storage=untyped_storage,
                dtype=self.dtype,
                _internal=True)
        else:
            return type(self)(wrap_storage=untyped_storage)

    def __len__(self):
        _warn_typed_storage_removal()
        return self._size()

    def _maybe_wrap_index(self, idx, is_stop=False):
        if idx is None:
            if is_stop:
                return self._size()
            else:
                return 0

        else:
            if type(idx) != int:
                raise TypeError(
                    f"can't index a {type(self)} with {type(idx)}")
            if is_stop:
                if (idx > self._size()) or (idx < -self._size()):
                    raise IndexError(
                        f'index {idx} out of range for storage of size {self.size()}')
                if idx > 0:
                    return idx
                else:
                    return idx % self._size()
            else:
                if (idx >= self._size()) or (idx < -self._size()):
                    raise IndexError(
                        f'index {idx} out of range for storage of size {self.size()}')
                return idx % self._size()

    def __setitem__(self, idx, value):
        _warn_typed_storage_removal()
        return self._setitem(idx, value)

    def _setitem(self, idx, value):
        if not isinstance(idx, (int, slice)):
            raise RuntimeError(f"can't index a {type(self)} with {type(idx)}")
        if torch.is_storage(value):
            raise RuntimeError(f'cannot set item with value type {type(value)}')
        if self.dtype in [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]:
            interpret_dtypes = {
                torch.quint8: torch.uint8,
                torch.quint4x2: torch.uint8,
                torch.quint2x4: torch.uint8,
                torch.qint32: torch.int32,
                torch.qint8: torch.int8
            }
            tmp_dtype = interpret_dtypes[self.dtype]
            tmp_tensor = torch.tensor([], dtype=tmp_dtype, device=self._untyped_storage.device)
            tmp_tensor.set_(TypedStorage(
                wrap_storage=self._untyped_storage,
                dtype=tmp_dtype,
                _internal=True))
        else:
            tmp_tensor = torch.tensor([], dtype=self.dtype, device=self._untyped_storage.device).set_(self)

        tmp_tensor[idx] = value

    def __getitem__(self, idx):
        _warn_typed_storage_removal()
        return self._getitem(idx)

    def _getitem(self, idx):
        if self._untyped_storage.device.type == 'meta':
            raise NotImplementedError("Not available for 'meta' device type")

        # NOTE: Before TypedStorage existed, indexing with a slice used to be
        # possible for <type>Storage objects. However, it would return
        # a storage view, which would be a hassle to implement in TypedStorage,
        # so it was disabled
        if isinstance(idx, slice):
            raise RuntimeError('slices are only supported in UntypedStorage.__getitem__')
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
            return TypedStorage(
                wrap_storage=self._untyped_storage,
                dtype=interpret_dtypes[self.dtype],
                _internal=True)._getitem(idx)

        idx_wrapped = self._maybe_wrap_index(idx)
        tmp_tensor = torch.tensor([], dtype=self.dtype, device=self._untyped_storage.device).set_(self)
        return tmp_tensor[idx_wrapped].item()

    def copy_(self, source: T, non_blocking: _Optional[bool] = None):
        _warn_typed_storage_removal()
        if isinstance(source, TypedStorage):
            self._untyped_storage.copy_(source._untyped_storage, non_blocking)  # type: ignore[arg-type]
        else:
            self._untyped_storage.copy_(source, non_blocking)  # type: ignore[arg-type]
        return self

    def nbytes(self):
        _warn_typed_storage_removal()
        return self._nbytes()

    # For internal use only, to avoid deprecation warning
    def _nbytes(self):
        return self._untyped_storage.nbytes()

    def type(self, dtype: _Optional[str] = None, non_blocking: bool = False) -> Union[T, str]:
        _warn_typed_storage_removal()
        if dtype is None:
            legacy_class = self._get_legacy_storage_class()

            if legacy_class is not None:
                return legacy_class.__module__ + '.' + legacy_class.__name__

            return '.'.join([self.__module__, type(self).__name__])

        else:
            return self._untyped_storage.type(dtype, non_blocking)

    def cuda(self, device=None, non_blocking=False, **kwargs) -> T:  # type: ignore[misc, type-var]
        _warn_typed_storage_removal()
        if self.dtype in [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]:
            raise RuntimeError("Cannot create CUDA storage with quantized dtype")
        cuda_storage: torch.UntypedStorage = self._untyped_storage.cuda(device, non_blocking, **kwargs)
        return self._new_wrapped_storage(cuda_storage)

    def hpu(self, device=None, non_blocking=False, **kwargs) -> T:  # type: ignore[misc, type-var]
        _warn_typed_storage_removal()
        if self.dtype in [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]:
            raise RuntimeError("Cannot create HPU storage with quantized dtype")
        hpu_storage: torch.UntypedStorage = self._untyped_storage.hpu(device, non_blocking, **kwargs)
        return self._new_wrapped_storage(hpu_storage)

    def element_size(self):
        _warn_typed_storage_removal()
        return self._element_size()

    # For internal use only, to avoid deprecation warning
    def _element_size(self):
        return torch._utils._element_size(self.dtype)

    def get_device(self) -> int:
        _warn_typed_storage_removal()
        return self._untyped_storage.get_device()

    def __str__(self):
        _warn_typed_storage_removal()
        info_str = (
            f'[{torch.typename(self)}(dtype={self.dtype}, '
            f'device={self.device}) of size {len(self)}]')
        if self.device.type == 'meta':
            return '...\n' + info_str
        else:
            data_str = ' ' + '\n '.join(str(self[i]) for i in range(self.size()))
            return data_str + '\n' + info_str

    def __repr__(self):
        _warn_typed_storage_removal()
        return str(self)

    def __iter__(self):
        _warn_typed_storage_removal()
        return iter(self[i] for i in range(self.size()))

    def __copy__(self):
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(copy.copy(self._untyped_storage))

    def __deepcopy__(self, memo):
        _warn_typed_storage_removal()
        return self._deepcopy(memo)

    # For internal use only, to avoid deprecation warning
    def _deepcopy(self, memo):
        return self._new_wrapped_storage(copy.deepcopy(self._untyped_storage, memo))

    def __sizeof__(self):
        _warn_typed_storage_removal()
        return super().__sizeof__() + self.nbytes()

    def clone(self):
        """Returns a copy of this storage"""
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(self._untyped_storage.clone())

    def tolist(self):
        """Returns a list containing the elements of this storage"""
        _warn_typed_storage_removal()
        return list(self)

    def cpu(self):
        """Returns a CPU copy of this storage if it's not already on the CPU"""
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(self._untyped_storage.cpu())

    def is_pinned(self, device: Union[str, torch.device] = 'cuda'):
        r"""Determine whether the CPU TypedStorage is already pinned on device.

        Args:
            device (str or torch.device): The device to pin memory on. Default: ``'cuda'``

        Returns:
            A boolean variable.
        """
        _warn_typed_storage_removal()
        return self._untyped_storage.is_pinned(device)

    def pin_memory(self, device: Union[str, torch.device] = 'cuda'):
        r"""Copies the CPU TypedStorage to pinned memory, if it's not already pinned.

        Args:
            device (str or torch.device): The device to pin memory on. Default: ``'cuda'``.

        Returns:
            A pinned CPU storage.
        """
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(self._untyped_storage.pin_memory(device=device))

    def share_memory_(self):
        """Moves the storage to shared memory.

        This is a no-op for storages already in shared memory and for CUDA
        storages, which do not need to be moved for sharing across processes.
        Storages in shared memory cannot be resized.

        Returns: self
        """
        _warn_typed_storage_removal()
        return self._share_memory_()

    # For internal use only, to avoid deprecation warning
    def _share_memory_(self):
        self._untyped_storage.share_memory_()
        return self

    def _new_shared(self, size, *, device=None):
        """Creates a new storage in shared memory with the same data type"""
        if device is None:
            device = 'cpu'
        device = torch.device(device)
        untyped_storage = torch.UntypedStorage._new_shared(size * self._element_size(), device=device)
        return TypedStorage(
            wrap_storage=untyped_storage,
            dtype=self.dtype,
            _internal=True)

    @property
    def _cdata(self):
        return self._untyped_storage._cdata

    @property
    def device(self):
        _warn_typed_storage_removal()
        return self._untyped_storage.device

    def size(self):
        _warn_typed_storage_removal()
        return self._size()

    # For internal use only, to avoid deprecation warning
    def _size(self):
        # NB: don't indirect through __len__, as that requires
        # an int to be returned
        return self._untyped_storage.nbytes() // self._element_size()

    def pickle_storage_type(self):
        _warn_typed_storage_removal()
        return self._pickle_storage_type()

    # For internal use only, to avoid deprecation warning
    def _pickle_storage_type(self):
        try:
            return _dtype_to_storage_type_map()[self.dtype]
        except KeyError as e:
            raise KeyError(f'dtype {self.dtype} is not recognized') from e

    def __reduce__(self):
        b = io.BytesIO()
        torch.save(self, b, _use_new_zipfile_serialization=False)
        return (_load_from_bytes, (b.getvalue(),))

    def data_ptr(self):
        _warn_typed_storage_removal()
        return self._data_ptr()

    # For internal use only, to avoid deprecation warning
    def _data_ptr(self):
        return self._untyped_storage.data_ptr()

    def resize_(self, size):
        _warn_typed_storage_removal()
        self._resize_(size)

    # For internal use only, to avoid deprecation warning
    def _resize_(self, size):
        self._untyped_storage.resize_(size * self._element_size())

    @classmethod
    def _free_weak_ref(cls, *args, **kwargs):
        return UntypedStorage._free_weak_ref(*args, **kwargs)

    def _weak_ref(self, *args, **kwargs):
        return self._untyped_storage._weak_ref(*args, **kwargs)

    @classmethod
    def from_buffer(cls, *args, **kwargs):
        _warn_typed_storage_removal()
        return cls._from_buffer(*args, **kwargs)

    @classmethod
    def _from_buffer(cls, *args, dtype=None, device=None, **kwargs):
        if cls == TypedStorage:
            dtype = torch.get_default_dtype() if dtype is None else dtype
            device = torch.device('cpu' if device is None else device)
            if device.type != 'cpu':
                raise RuntimeError(f'TypedStorage.from_buffer: Not available for device {device.type}')
            untyped_storage: torch.UntypedStorage = torch.UntypedStorage.from_buffer(*args, dtype=dtype, **kwargs)

        else:
            if dtype is not None or len(args) == 5:
                raise RuntimeError(
                    "from_buffer: 'dtype' can only be specified in "
                    "UntypedStorage.from_buffer and TypedStorage.from_buffer")
            if device is not None:
                raise RuntimeError(
                    "from_buffer: 'device' can only be specified in "
                    "UntypedStorage.from_buffer and TypedStorage.from_buffer")

            dtype = cls._dtype
            untyped_storage = torch.UntypedStorage.from_buffer(*args, dtype=dtype, **kwargs)

        return TypedStorage(
            wrap_storage=untyped_storage,
            dtype=dtype,
            _internal=True)

    def _to(self, dtype):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Argument 'dtype' must be torch.dtype, not {type(dtype)}")
        storage = torch.tensor([], dtype=self.dtype, device=self.device).set_(self).to(dtype)._typed_storage()
        if storage.data_ptr() == self.data_ptr():
            storage = storage.clone()
        return storage

    def double(self):
        """Casts this storage to double type"""
        _warn_typed_storage_removal()
        return self._to(torch.double)

    def float(self):
        """Casts this storage to float type"""
        _warn_typed_storage_removal()
        return self._to(torch.float)

    def half(self):
        """Casts this storage to half type"""
        _warn_typed_storage_removal()
        return self._to(torch.half)

    def long(self):
        """Casts this storage to long type"""
        _warn_typed_storage_removal()
        return self._to(torch.long)

    def int(self):
        """Casts this storage to int type"""
        _warn_typed_storage_removal()
        return self._to(torch.int)

    def short(self):
        """Casts this storage to short type"""
        _warn_typed_storage_removal()
        return self._to(torch.short)

    def char(self):
        """Casts this storage to char type"""
        _warn_typed_storage_removal()
        return self._to(torch.int8)

    def byte(self):
        """Casts this storage to byte type"""
        _warn_typed_storage_removal()
        return self._to(torch.uint8)

    def bool(self):
        """Casts this storage to bool type"""
        _warn_typed_storage_removal()
        return self._to(torch.bool)

    def bfloat16(self):
        """Casts this storage to bfloat16 type"""
        _warn_typed_storage_removal()
        return self._to(torch.bfloat16)

    def complex_double(self):
        """Casts this storage to complex double type"""
        _warn_typed_storage_removal()
        return self._to(torch.cdouble)

    def complex_float(self):
        """Casts this storage to complex float type"""
        _warn_typed_storage_removal()
        return self._to(torch.cfloat)

    @classmethod
    def from_file(cls, filename, shared, size):
        """
        from_file(filename, shared=False, size=0) -> Storage

        If `shared` is `True`, then memory is shared between all processes.
        All changes are written to the file. If `shared` is `False`, then the changes on
        the storage do not affect the file.

        `size` is the number of elements in the storage. If `shared` is `False`,
        then the file must contain at least `size * sizeof(Type)` bytes
        (`Type` is the type of storage). If `shared` is `True` the file will be
        created if needed.

        Args:
            filename (str): file name to map
            shared (bool): whether to share memory
            size (int): number of elements in the storage
        """
        _warn_typed_storage_removal()
        if cls == TypedStorage:
            raise RuntimeError('from_file can only be called on derived classes')
        untyped_storage: UntypedStorage = UntypedStorage.from_file(
            filename,
            shared,
            size * torch._utils._element_size(cls.dtype))
        storage = cls(wrap_storage=untyped_storage)
        return storage

    @classmethod
    def _expired(cls, *args, **kwargs):
        return UntypedStorage._expired(*args, **kwargs)

    def _write_file(self, *args, **kwargs):
        return self._untyped_storage._write_file(*args, **kwargs)

    def _set_from_file(self, *args, **kwargs):
        return self._untyped_storage._set_from_file(*args, **kwargs)

    def _set_cdata(self, *args, **kwargs):
        return self._untyped_storage._set_cdata(*args, **kwargs)

    def _share_cuda_(self, *args, **kwargs):
        return self._untyped_storage._share_cuda_(*args, **kwargs)

    def is_shared(self):
        _warn_typed_storage_removal()
        return self._is_shared()

    # For internal use only, to avoid deprecation warning
    def _is_shared(self):
        return self._untyped_storage.is_shared()

    @classmethod
    def _new_shared_cuda(cls, *args, **kwargs):
        return torch.UntypedStorage._new_shared_cuda(*args, **kwargs)

    def _share_filename_cpu_(self, *args, **kwargs):
        manager_handle, storage_handle, size = self._untyped_storage._share_filename_cpu_(*args, **kwargs)
        return manager_handle, storage_handle, size // self._element_size()

    def _shared_decref(self):
        self._untyped_storage._shared_decref()
        return self

    @classmethod
    def _release_ipc_counter(cls, *args, device=None, **kwargs):
        return torch.UntypedStorage._release_ipc_counter_cuda(*args, **kwargs)

    def _shared_incref(self, *args, **kwargs):
        return self._untyped_storage._shared_incref(*args, **kwargs)

    def _share_fd_cpu_(self, *args, **kwargs):
        fd, size = self._untyped_storage._share_fd_cpu_(*args, **kwargs)
        return fd, size // self._element_size()

    def _get_legacy_storage_class(self):
        if self.dtype not in _dtype_to_storage_type_map():
            return None

        storage_name = _dtype_to_storage_type_map()[self.dtype]

        if self.device.type not in ['cpu', 'cuda', torch._C._get_privateuse1_backend_name()]:
            return None

        module = torch if self.device.type == 'cpu' else getattr(torch, self.device.type)

        try:
            return getattr(module, storage_name)
        except AttributeError:
            return None

TypedStorage.type.__doc__ = _type.__doc__
TypedStorage.cuda.__doc__ = _cuda.__doc__
TypedStorage.hpu.__doc__ = _hpu.__doc__

class _LegacyStorageMeta(type):
    dtype: torch.dtype

    def __instancecheck__(cls, instance):
        if type(instance) == TypedStorage:
            cls_device = _get_device_from_module(cls.__module__)
            return (cls_device == instance.device.type) and (cls.dtype == instance.dtype)
        return False

class _LegacyStorage(TypedStorage, metaclass=_LegacyStorageMeta):
    @classmethod
    def _new_shared(cls, size):
        """Creates a new storage in shared memory with the same data type"""
        untyped_storage = torch.UntypedStorage._new_shared(size * cls()._element_size())
        return cls(wrap_storage=untyped_storage)

    @classmethod
    def _release_ipc_counter(cls, *args, **kwargs):
        return torch.UntypedStorage._release_ipc_counter_cuda(*args, **kwargs)

    @classmethod
    def _new_shared_filename(cls, manager, obj, size):
        bytes_size = size * torch._utils._element_size(cls.dtype)
        return cls(wrap_storage=torch.UntypedStorage._new_shared_filename_cpu(manager, obj, bytes_size))

def _get_dtype_from_pickle_storage_type(pickle_storage_type: str):
    try:
        return _storage_type_to_dtype_map()[pickle_storage_type]
    except KeyError as e:
        raise KeyError(
            f'pickle storage type "{pickle_storage_type}" is not recognized') from e
