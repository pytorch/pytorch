# mypy: allow-untyped-defs

from __future__ import annotations

import collections
import copy
import functools
import io
import threading
import warnings
from typing import Any, cast, Optional as _Optional, TYPE_CHECKING, TypeVar, Union
from typing_extensions import Self

import torch
from torch._utils import _to, _type
from torch.types import _bool, _int, Storage


if TYPE_CHECKING:
    from torch._prims_common import DeviceLikeType


__all__ = ["TypedStorage", "UntypedStorage"]


try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]


_share_memory_lock = threading.Lock()
_share_memory_map: dict[int, threading.RLock] = {}

T = TypeVar("T", bound="Union[_StorageBase, TypedStorage]")


class _StorageBase:
    _cdata: Any
    is_sparse: _bool = False
    is_sparse_csr: _bool = False
    device: torch.device
    # Used when
    # (1) stashing FakeTensor device onto storage in torch.serialization.skip_data
    # (2) stashing device onto storage to propagate to FakeTensor when torch.load under FakeTensorMode
    _fake_device: _Optional[torch.device] = None
    # Used when loading with FakeTensorMode to give information about offset of storage in torch.saved-file
    _checkpoint_offset: _Optional[int] = None

    def __init__(self, *args, **kwargs):
        pass

    def __len__(self) -> _int:
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __setitem__(self, *args, **kwargs):
        raise NotImplementedError

    def copy_(self, source: T, non_blocking: _Optional[_bool] = None) -> T:
        raise NotImplementedError

    def new(self) -> Union[_StorageBase, TypedStorage]:
        raise NotImplementedError

    def nbytes(self) -> _int:
        raise NotImplementedError

    def size(self) -> _int:
        return self.nbytes()

    def type(
        self, dtype: _Optional[str] = None, non_blocking: _bool = False
    ) -> Union[_StorageBase, TypedStorage]:
        return _type(self, dtype, non_blocking)

    def cuda(
        self, device=None, non_blocking=False
    ) -> Union[_StorageBase, TypedStorage]:
        """Returns a copy of this object in CUDA memory.

        If this object is already in CUDA memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination GPU id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
        """
        device2 = torch.device("cuda", device) if device else torch.device("cuda")
        return self.to(device=device2, non_blocking=non_blocking)

    def hpu(self, device=None, non_blocking=False) -> Union[_StorageBase, TypedStorage]:
        """Returns a copy of this object in HPU memory.

        If this object is already in HPU memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination HPU id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
        """
        device2 = torch.device("hpu", device) if device else torch.device("hpu")
        return self.to(device=device2, non_blocking=non_blocking)

    def element_size(self) -> _int:
        raise NotImplementedError

    def get_device(self) -> _int:
        return self.device.index

    def data_ptr(self) -> _int:
        raise NotImplementedError

    def resizable(self) -> _bool:
        raise NotImplementedError

    # Defined in torch/csrc/generic/StorageSharing.cpp
    def _share_filename_cpu_(self, *args, **kwargs):
        raise NotImplementedError

    def _share_fd_cpu_(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def _new_using_filename_cpu(cls, size: _int) -> Self:
        raise NotImplementedError

    @classmethod
    def _new_using_fd_cpu(cls, size: _int) -> Self:
        raise NotImplementedError

    @classmethod
    def from_buffer(cls, *args, **kwargs) -> Self:
        raise NotImplementedError

    @classmethod
    def _new_shared_filename_cpu(
        cls,
        manager,
        obj,
        size,
        *,
        device=None,
        dtype=None,
    ) -> Self:
        raise NotImplementedError

    @classmethod
    def _release_ipc_counter_cuda(cls, *args, **kwargs) -> Self:
        raise NotImplementedError

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs) -> Self:
        raise NotImplementedError

    def _shared_decref(self) -> Union[_StorageBase, TypedStorage]:
        raise NotImplementedError

    def _write_file(self, *args, **kwargs):
        raise NotImplementedError

    def resize_(self, size: _int):
        raise NotImplementedError

    def _weak_ref(self, *args, **kwargs) -> Union[_StorageBase, TypedStorage]:
        raise NotImplementedError

    def _set_from_file(self, *args, **kwargs):
        raise NotImplementedError

    def _set_cdata(self, *args, **kwargs):
        raise NotImplementedError

    def _share_cuda_(self, *args, **kwargs):
        raise NotImplementedError

    def is_shared(self) -> _bool:
        raise NotImplementedError

    @classmethod
    def _new_shared_cuda(cls, *args, **kwargs) -> Self:
        raise NotImplementedError

    def _shared_incref(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def _free_weak_ref(cls, *args, **kwargs):
        raise NotImplementedError

    @property
    def is_cuda(self):
        raise NotImplementedError

    @property
    def is_hpu(self):
        raise NotImplementedError

    @classmethod
    def from_file(cls, filename, shared, nbytes) -> Union[_StorageBase, TypedStorage]:
        raise NotImplementedError

    @classmethod
    def _expired(cls, *args, **kwargs) -> Union[_StorageBase, TypedStorage]:
        raise NotImplementedError

    def _byteswap(self, *args, **kwargs):
        raise NotImplementedError

    def _get_filename(self, *args, **kwargs) -> _Optional[str]:
        raise NotImplementedError

    def __repr__(self):
        info_str = f"[{torch.typename(self)}(device={self.device}) of size {len(self)}]"
        if self.device.type == "meta":
            return "...\n" + info_str
        data_str = " " + "\n ".join(str(self[i]) for i in range(self.size()))
        return data_str + "\n" + info_str

    def __iter__(self):
        return iter(self[i] for i in range(self.size()))

    def __copy__(self):
        return self.clone()

    def __deepcopy__(self, memo):
        memo = memo.setdefault("torch", {})
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
        """Return a copy of this storage."""
        return type(self)(self.nbytes(), device=self.device).copy_(self)

    def tolist(self):
        """Return a list containing the elements of this storage."""
        return list(self)

    def cpu(self):
        """Return a CPU copy of this storage if it's not already on the CPU."""
        if self.device.type != "cpu":
            return torch.UntypedStorage(self.size()).copy_(self, False)
        return self

    def mps(self):
        """Return a MPS copy of this storage if it's not already on the MPS."""
        if self.device.type != "mps":
            return torch.UntypedStorage(self.size(), device="mps").copy_(self, False)
        return self

    def _to(self, dtype):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Argument 'dtype' must be torch.dtype, not {type(dtype)}")
        storage = (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .to(dtype)
            ._typed_storage()
        )
        if storage.data_ptr() == self.data_ptr():
            storage = storage.clone()
        return storage

    def to(self, *, device: DeviceLikeType, non_blocking: _bool = False):
        if not isinstance(device, torch.device):
            device = torch.device(device)
        return _to(self, device, non_blocking)

    def double(self):
        """Casts this storage to double type."""
        return self._to(torch.double)

    def float(self):
        """Casts this storage to float type."""
        return self._to(torch.float)

    def half(self):
        """Casts this storage to half type."""
        return self._to(torch.half)

    def long(self):
        """Casts this storage to long type."""
        return self._to(torch.long)

    def int(self):
        """Casts this storage to int type."""
        return self._to(torch.int)

    def short(self):
        """Casts this storage to short type."""
        return self._to(torch.short)

    def char(self):
        """Casts this storage to char type."""
        return self._to(torch.int8)

    def byte(self):
        """Casts this storage to byte type."""
        return self._to(torch.uint8)

    def bool(self):
        """Casts this storage to bool type."""
        return self._to(torch.bool)

    def bfloat16(self):
        """Casts this storage to bfloat16 type."""
        return self._to(torch.bfloat16)

    def complex_double(self):
        """Casts this storage to complex double type."""
        return self._to(torch.cdouble)

    def complex_float(self):
        """Casts this storage to complex float type."""
        return self._to(torch.cfloat)

    def float8_e5m2(self):
        """Casts this storage to float8_e5m2 type"""
        return self._to(torch.float8_e5m2)

    def float8_e4m3fn(self):
        """Casts this storage to float8_e4m3fn type"""
        return self._to(torch.float8_e4m3fn)

    def float8_e5m2fnuz(self):
        """Casts this storage to float8_e5m2fnuz type"""
        return self._to(torch.float8_e5m2fnuz)

    def float8_e4m3fnuz(self):
        """Casts this storage to float8_e4m3fnuz type"""
        return self._to(torch.float8_e4m3fnuz)

    def is_pinned(self, device: Union[str, torch.device] = "cuda"):
        r"""Determine whether the CPU storage is already pinned on device.

        Args:
            device (str or torch.device): The device to pin memory on (default: ``'cuda'``).
                This argument is discouraged and subject to deprecated.

        Returns:
            A boolean variable.
        """
        return (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .is_pinned(device)
        )

    def pin_memory(self, device: Union[str, torch.device] = "cuda"):
        r"""Copy the CPU storage to pinned memory, if it's not already pinned.

        Args:
            device (str or torch.device): The device to pin memory on (default: ``'cuda'``).
                This argument is discouraged and subject to deprecated.

        Returns:
            A pinned CPU storage.
        """
        if self.device.type != "cpu":
            raise TypeError(f"cannot pin '{self.type()}' only CPU memory can be pinned")

        pinned_tensor = (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .pin_memory(device)
        )
        return pinned_tensor.untyped_storage()

    def share_memory_(self):
        """See :meth:`torch.UntypedStorage.share_memory_`"""
        from torch.multiprocessing import get_sharing_strategy

        if self.device.type in ["cuda", torch._C._get_privateuse1_backend_name()]:
            pass  # CUDA or PrivateUse1 doesn't use POSIX shared memory
        elif get_sharing_strategy() == "file_system":
            self._share_filename_cpu_()
        else:
            self._share_fd_cpu_()
        return self

    @classmethod
    def _new_shared(cls, size, *, device="cpu"):
        """Create a new storage in shared memory with the same data type."""
        from torch.multiprocessing import get_sharing_strategy

        device = torch.device(device)
        if device.type in ["cuda", torch._C._get_privateuse1_backend_name(), "hpu"]:
            return cls(size, device=device)
        elif get_sharing_strategy() == "file_system":
            return cls._new_using_filename_cpu(size)
        else:
            return cls._new_using_fd_cpu(size)

    def untyped(self):
        return self

    def byteswap(self, dtype):
        """Swap bytes in underlying data."""
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
        if self.device.type == "meta":
            raise NotImplementedError("Not available for 'meta' device type")
        return super().__getitem__(*args, **kwargs)

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    @property
    def is_hpu(self):
        return self.device.type == "hpu"

    @property
    def filename(self) -> _Optional[str]:
        """Returns the file name associated with this storage.

        The file name will be a string if the storage is on CPU and was created via
        :meth:`~torch.from_file()` with ``shared`` as ``True``. This attribute is ``None`` otherwise.
        """
        return self._get_filename()

    @_share_memory_lock_protected
    def share_memory_(self, *args, **kwargs):
        """
        Moves the storage to shared memory.

        This is a no-op for storages already in shared memory and for CUDA
        storages, which do not need to be moved for sharing across processes.
        Storages in shared memory cannot be resized.

        Note that to mitigate issues like `this <https://github.com/pytorch/pytorch/issues/95606>`_
        it is thread safe to call this function from multiple threads on the same object.
        It is NOT thread safe though to call any other function on self without proper
        synchronization. Please see :doc:`/notes/multiprocessing` for more details.

        .. note::
            When all references to a storage in shared memory are deleted, the associated shared memory
            object will also be deleted. PyTorch has a special cleanup process to ensure that this happens
            even if the current process exits unexpectedly.

            It is worth noting the difference between :meth:`share_memory_` and :meth:`from_file` with ``shared = True``

            #. ``share_memory_`` uses `shm_open(3) <https://man7.org/linux/man-pages/man3/shm_open.3.html>`_ to create a
               POSIX shared memory object while :meth:`from_file` uses
               `open(2) <https://man7.org/linux/man-pages/man2/open.2.html>`_ to open the filename passed by the user.
            #. Both use an `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_ with ``MAP_SHARED``
               to map the file/object into the current virtual address space
            #. ``share_memory_`` will call ``shm_unlink(3)`` on the object after mapping it to make sure the shared memory
               object is freed when no process has the object open. ``torch.from_file(shared=True)`` does not unlink the
               file. This file is persistent and will remain until it is deleted by the user.

        Returns:
            ``self``
        """
        return super().share_memory_(*args, **kwargs)

    @_share_memory_lock_protected
    def _share_fd_cpu_(self, *args, **kwargs):
        return super()._share_fd_cpu_(*args, **kwargs)

    @_share_memory_lock_protected
    def _share_filename_cpu_(self, *args, **kwargs):
        return super()._share_filename_cpu_(*args, **kwargs)


def _load_from_bytes(b):
    return torch.load(io.BytesIO(b), weights_only=False)


@functools.cache
def _new_dtypes():
    # These are dtypes serialized as UntypedStorage unlike those in
    # _dtype_to_storage_type_map
    return {
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fnuz,
        torch.float8_e8m0fnu,
        torch.bits8,
        torch.bits16,
        torch.bits1x8,
        torch.bits2x4,
        torch.bits4x2,
        torch.complex32,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    }


@functools.cache
def _dtype_to_storage_type_map():
    # NOTE: We should no longer add dtypes to this map. This map
    # is only used for BC/FC with older PyTorch versions. Going forward,
    # new dtypes of TypedStorage should not translate to a legacy
    # <type>Storage class. Instead, new dtypes of TypedStorage should
    # be serialized as an UntypedStorage paired with a torch.dtype
    return {
        torch.double: "DoubleStorage",
        torch.float: "FloatStorage",
        torch.half: "HalfStorage",
        torch.long: "LongStorage",
        torch.int: "IntStorage",
        torch.int16: "ShortStorage",
        torch.int8: "CharStorage",
        torch.uint8: "ByteStorage",
        torch.bool: "BoolStorage",
        torch.bfloat16: "BFloat16Storage",
        torch.cdouble: "ComplexDoubleStorage",
        torch.cfloat: "ComplexFloatStorage",
        torch.qint8: "QInt8Storage",
        torch.qint32: "QInt32Storage",
        torch.quint8: "QUInt8Storage",
        torch.quint4x2: "QUInt4x2Storage",
        torch.quint2x4: "QUInt2x4Storage",
    }


@functools.cache
def _storage_type_to_dtype_map():
    dtype_map = {val: key for key, val in _dtype_to_storage_type_map().items()}
    return dtype_map


def _get_storage_from_sequence(sequence, dtype, device):
    if dtype in [
        torch.quint8,
        torch.quint4x2,
        torch.quint2x4,
        torch.qint32,
        torch.qint8,
    ]:
        interpret_dtypes = {
            torch.quint8: torch.uint8,
            torch.quint4x2: torch.uint8,
            torch.quint2x4: torch.uint8,
            torch.qint32: torch.int32,
            torch.qint8: torch.int8,
        }
        tmp_tensor = torch.tensor(
            sequence, dtype=interpret_dtypes[dtype], device=device
        )

    else:
        tmp_tensor = torch.tensor(sequence, dtype=dtype, device=device)

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
        if not hasattr(_warn_typed_storage_removal, "has_warned"):
            return True
        else:
            return not _warn_typed_storage_removal.__dict__["has_warned"]

    if _get_always_warn_typed_storage_removal() or is_first_time():
        message = (
            "TypedStorage is deprecated. It will be removed in the future and "
            "UntypedStorage will be the only storage class. This should only matter "
            "to you if you are using storages directly.  To access UntypedStorage "
            "directly, use tensor.untyped_storage() instead of tensor.storage()"
        )
        warnings.warn(message, UserWarning, stacklevel=stacklevel + 1)
        _warn_typed_storage_removal.__dict__["has_warned"] = True


def _reset_warn_typed_storage_removal():
    _warn_typed_storage_removal.__dict__["has_warned"] = False


def _get_device_from_module(module: str):
    last_part = module.rsplit(".", 1)[-1]
    if last_part in ["cuda", torch._C._get_privateuse1_backend_name(), "hpu"]:
        return last_part
    else:
        return "cpu"


class TypedStorage:
    is_sparse: _bool = False
    # Used when stashing FakeTensor device onto storage in torch.save(metadata_only=True)
    _fake_device: _Optional[torch.device] = None

    dtype: torch.dtype

    @property
    def _dtype(self):
        return self.dtype

    @property
    def filename(self) -> _Optional[str]:
        """Returns the file name associated with this storage if the storage was memory mapped from a file.
        or ``None`` if the storage was not created by memory mapping a file."""
        return self._untyped_storage.filename

    def fill_(self, value):
        _warn_typed_storage_removal()
        self._setitem(slice(0, self._size()), value)
        return self

    def __new__(
        cls,
        *args,
        wrap_storage=None,
        dtype=None,
        device=None,
        _internal=False,
    ):
        if not _internal:
            _warn_typed_storage_removal()

        if cls == torch.storage._LegacyStorage:
            raise RuntimeError(
                "Only child classes of _LegacyStorage can be instantiated"
            )

        if cls == TypedStorage:
            return super().__new__(cls)

        else:
            arg_error_msg = (
                f"{cls}.__new__ received an invalid combination "
                f"of arguments. Expected one of:\n"
                " * no arguments\n"
                " * (int size)\n"
                " * (Sequence data)\n"
                " * (*, UntypedStorage wrap_storage)"
            )

            if device is not None:
                raise RuntimeError(
                    arg_error_msg + "\nKeyword argument 'device' cannot be specified"
                )

            if dtype is not None:
                raise RuntimeError(
                    arg_error_msg + "\nKeyword argument 'dtype' cannot be specified"
                )

            if wrap_storage is None:
                if len(args) > 1:
                    raise RuntimeError(
                        arg_error_msg + "\nToo many positional arguments"
                    )

                if (
                    len(args) == 1
                    and not _isint(args[0])
                    and not isinstance(args[0], collections.abc.Sequence)
                ):
                    raise TypeError(
                        arg_error_msg
                        + f"\nArgument type not recognized: {type(args[0])}"
                    )

                return TypedStorage(
                    *args,
                    dtype=cls._dtype,
                    device=_get_device_from_module(cls.__module__),
                    _internal=True,
                )

            else:
                if len(args) != 0:
                    raise RuntimeError(
                        arg_error_msg
                        + "\nNo positional arguments should be given when using "
                        "'wrap_storage'"
                    )

                if not isinstance(wrap_storage, torch.UntypedStorage):
                    raise TypeError(
                        arg_error_msg
                        + f"\nArgument 'wrap_storage' must be UntypedStorage, but got {type(wrap_storage)}"
                    )

                cls_device = _get_device_from_module(cls.__module__)

                if wrap_storage.device.type != cls_device:
                    raise RuntimeError(
                        arg_error_msg
                        + f"\nDevice of 'wrap_storage' must be {cls_device}"
                        f", but got {wrap_storage.device.type}"
                    )

                return TypedStorage(
                    *args,
                    wrap_storage=wrap_storage,
                    dtype=cls.dtype,
                    _internal=True,
                )

    def __init__(
        self,
        *args,
        device=None,
        dtype=None,
        wrap_storage=None,
        _internal=False,
    ):
        if not _internal:
            _warn_typed_storage_removal()
        arg_error_msg = (
            "TypedStorage.__init__ received an invalid combination "
            "of arguments. Expected one of:\n"
            " * (*, torch.device device, torch.dtype dtype)\n"
            " * (int size, *, torch.device device, torch.dtype dtype)\n"
            " * (Sequence data, *, torch.device device, torch.dtype dtype)\n"
            " * (*, UntypedStorage wrap_storage, torch.dtype dtype)"
        )

        if wrap_storage is not None:
            if len(args) != 0:
                raise RuntimeError(
                    arg_error_msg
                    + "\nNo positional arguments should be given when using "
                    "'wrap_storage'"
                )

            if dtype is None:
                raise RuntimeError(
                    arg_error_msg + "\nArgument 'dtype' must be specified"
                )

            if not isinstance(dtype, torch.dtype):
                raise TypeError(
                    arg_error_msg
                    + f"\nArgument 'dtype' must be torch.dtype, not {type(dtype)}"
                )

            if device is not None:
                raise RuntimeError(
                    arg_error_msg
                    + "\nArgument 'device' should not be specified when 'wrap_storage' is given"
                )

            self.dtype = dtype

            if not isinstance(wrap_storage, torch.UntypedStorage):
                raise TypeError(
                    arg_error_msg
                    + f"\nArgument 'wrap_storage' must be UntypedStorage, but got {type(wrap_storage)}"
                )

            self._untyped_storage = wrap_storage

        else:
            self.dtype = torch.get_default_dtype() if dtype is None else dtype
            device = torch.device("cpu" if device is None else device)

            if self.dtype in [
                torch.quint8,
                torch.quint4x2,
                torch.quint2x4,
                torch.qint32,
                torch.qint8,
            ]:
                if device.type == "cuda":
                    raise RuntimeError(
                        "Cannot create CUDA storage with quantized dtype"
                    )

            if len(args) == 0:
                self._untyped_storage = torch.UntypedStorage(device=device)

            elif len(args) == 1:
                if _isint(args[0]):
                    self._untyped_storage = torch.UntypedStorage(
                        int(args[0]) * self._element_size(), device=device
                    )
                elif isinstance(args[0], collections.abc.Sequence):
                    self._untyped_storage = _get_storage_from_sequence(
                        args[0], self.dtype, device
                    )
                else:
                    raise TypeError(
                        arg_error_msg
                        + f"\nArgument type not recognized: {type(args[0])}"
                    )

            else:
                raise RuntimeError(arg_error_msg + "\nToo many positional arguments")

    @property
    def is_cuda(self):
        _warn_typed_storage_removal()
        return self._untyped_storage.device.type == "cuda"

    @property
    def is_hpu(self):
        _warn_typed_storage_removal()
        return self._untyped_storage.device.type == "hpu"

    def untyped(self):
        """Return the internal :class:`torch.UntypedStorage`."""
        _warn_typed_storage_removal()
        return self._untyped_storage

    def _new_wrapped_storage(self, untyped_storage) -> Self:
        assert type(untyped_storage) == torch.UntypedStorage

        if type(self) == TypedStorage:
            return cast(
                Self,
                TypedStorage(
                    wrap_storage=untyped_storage, dtype=self.dtype, _internal=True
                ),
            )
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
                raise TypeError(f"can't index a {type(self)} with {type(idx)}")
            if is_stop:
                if (idx > self._size()) or (idx < -self._size()):
                    raise IndexError(
                        f"index {idx} out of range for storage of size {self.size()}"
                    )
                if idx > 0:
                    return idx
                else:
                    return idx % self._size()
            else:
                if (idx >= self._size()) or (idx < -self._size()):
                    raise IndexError(
                        f"index {idx} out of range for storage of size {self.size()}"
                    )
                return idx % self._size()

    def __setitem__(self, idx, value):
        _warn_typed_storage_removal()
        return self._setitem(idx, value)

    def _setitem(self, idx, value):
        if not isinstance(idx, (int, slice)):
            raise RuntimeError(f"can't index a {type(self)} with {type(idx)}")
        if torch.is_storage(value):
            raise RuntimeError(f"cannot set item with value type {type(value)}")
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            interpret_dtypes = {
                torch.quint8: torch.uint8,
                torch.quint4x2: torch.uint8,
                torch.quint2x4: torch.uint8,
                torch.qint32: torch.int32,
                torch.qint8: torch.int8,
            }
            tmp_dtype = interpret_dtypes[self.dtype]
            tmp_tensor = torch.tensor(
                [], dtype=tmp_dtype, device=self._untyped_storage.device
            )
            tmp_tensor.set_(
                TypedStorage(
                    wrap_storage=self._untyped_storage, dtype=tmp_dtype, _internal=True
                )
            )
        else:
            tmp_tensor = torch.tensor(
                [], dtype=self.dtype, device=self._untyped_storage.device
            ).set_(self)

        tmp_tensor[idx] = value

    def __getitem__(self, idx):
        _warn_typed_storage_removal()
        return self._getitem(idx)

    def _getitem(self, idx):
        if self._untyped_storage.device.type == "meta":
            raise NotImplementedError("Not available for 'meta' device type")

        # NOTE: Before TypedStorage existed, indexing with a slice used to be
        # possible for <type>Storage objects. However, it would return
        # a storage view, which would be a hassle to implement in TypedStorage,
        # so it was disabled
        if isinstance(idx, slice):
            raise RuntimeError(
                "slices are only supported in UntypedStorage.__getitem__"
            )
        elif not isinstance(idx, int):
            raise RuntimeError(f"can't index a {type(self)} with {type(idx)}")

        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            interpret_dtypes = {
                torch.quint8: torch.uint8,
                torch.quint4x2: torch.uint8,
                torch.quint2x4: torch.uint8,
                torch.qint32: torch.int32,
                torch.qint8: torch.int8,
            }
            return TypedStorage(
                wrap_storage=self._untyped_storage,
                dtype=interpret_dtypes[self.dtype],
                _internal=True,
            )._getitem(idx)

        idx_wrapped = self._maybe_wrap_index(idx)
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        with unset_fake_temporarily():
            tmp_tensor = torch.tensor(
                [], dtype=self.dtype, device=self._untyped_storage.device
            ).set_(self)
            return tmp_tensor[idx_wrapped].item()

    def copy_(self, source: T, non_blocking: _Optional[bool] = None):
        _warn_typed_storage_removal()
        if isinstance(source, TypedStorage):
            self._untyped_storage.copy_(source._untyped_storage, non_blocking)
        else:
            self._untyped_storage.copy_(source, non_blocking)
        return self

    def nbytes(self):
        _warn_typed_storage_removal()
        return self._nbytes()

    # For internal use only, to avoid deprecation warning
    def _nbytes(self):
        return self._untyped_storage.nbytes()

    def type(
        self,
        dtype: _Optional[str] = None,
        non_blocking: bool = False,
    ) -> Union[_StorageBase, TypedStorage, str]:
        _warn_typed_storage_removal()
        if dtype is None:
            legacy_class = self._get_legacy_storage_class()

            if legacy_class is not None:
                return legacy_class.__module__ + "." + legacy_class.__name__

            return ".".join([self.__module__, type(self).__name__])

        else:
            return self._untyped_storage.type(dtype, non_blocking)

    def cuda(self, device=None, non_blocking=False) -> Self:
        _warn_typed_storage_removal()
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            raise RuntimeError("Cannot create CUDA storage with quantized dtype")
        cuda_storage = self._untyped_storage.cuda(device, non_blocking)
        return self._new_wrapped_storage(cuda_storage)

    def hpu(self, device=None, non_blocking=False) -> Self:
        _warn_typed_storage_removal()
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            raise RuntimeError("Cannot create HPU storage with quantized dtype")
        hpu_storage = self._untyped_storage.hpu(device, non_blocking)
        return self._new_wrapped_storage(hpu_storage)

    def to(self, *, device: DeviceLikeType, non_blocking: bool = False) -> Self:
        _warn_typed_storage_removal()
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            raise RuntimeError(
                f"Cannot create {device.type.upper()} storage with quantized dtype"
            )
        to_storage = self._untyped_storage.to(device=device, non_blocking=non_blocking)
        return self._new_wrapped_storage(to_storage)

    def element_size(self):
        _warn_typed_storage_removal()
        return self._element_size()

    # For internal use only, to avoid deprecation warning
    def _element_size(self):
        return torch._utils._element_size(self.dtype)

    def get_device(self) -> _int:
        _warn_typed_storage_removal()
        return self._untyped_storage.get_device()

    def __str__(self):
        _warn_typed_storage_removal()
        info_str = (
            f"[{torch.typename(self)}(dtype={self.dtype}, "
            f"device={self.device}) of size {len(self)}]"
        )
        if self.device.type == "meta":
            return "...\n" + info_str
        else:
            data_str = " " + "\n ".join(str(self[i]) for i in range(self.size()))
            return data_str + "\n" + info_str

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
        """Return a copy of this storage."""
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(self._untyped_storage.clone())

    def tolist(self):
        """Return a list containing the elements of this storage."""
        _warn_typed_storage_removal()
        return list(self)

    def cpu(self):
        """Return a CPU copy of this storage if it's not already on the CPU."""
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(self._untyped_storage.cpu())

    def is_pinned(self, device: Union[str, torch.device] = "cuda"):
        r"""Determine whether the CPU TypedStorage is already pinned on device.

        Args:
            device (str or torch.device): The device to pin memory on (default: ``'cuda'``).
                This argument is discouraged and subject to deprecated.

        Returns:
            A boolean variable.
        """
        _warn_typed_storage_removal()
        return self._untyped_storage.is_pinned(device)

    def pin_memory(self, device: Union[str, torch.device] = "cuda"):
        r"""Copy the CPU TypedStorage to pinned memory, if it's not already pinned.

        Args:
            device (str or torch.device): The device to pin memory on (default: ``'cuda'``).
                This argument is discouraged and subject to deprecated.

        Returns:
            A pinned CPU storage.
        """
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(
            self._untyped_storage.pin_memory(device=device)
        )

    def share_memory_(self):
        """See :meth:`torch.UntypedStorage.share_memory_`"""
        _warn_typed_storage_removal()
        return self._share_memory_()

    # For internal use only, to avoid deprecation warning
    def _share_memory_(self):
        self._untyped_storage.share_memory_()
        return self

    def _new_shared(self, size, *, device=None):
        """Create a new storage in shared memory with the same data type."""
        if device is None:
            device = "cpu"
        device = torch.device(device)
        untyped_storage = torch.UntypedStorage._new_shared(
            size * self._element_size(), device=device
        )
        return TypedStorage(
            wrap_storage=untyped_storage, dtype=self.dtype, _internal=True
        )

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
            raise KeyError(f"dtype {self.dtype} is not recognized") from e

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

    def resizable(self):
        _warn_typed_storage_removal()
        return self._untyped_storage.resizable()

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
            device = torch.device("cpu" if device is None else device)
            if device.type != "cpu":
                raise RuntimeError(
                    f"TypedStorage.from_buffer: Not available for device {device.type}"
                )
            untyped_storage: torch.UntypedStorage = torch.UntypedStorage.from_buffer(
                *args, dtype=dtype, **kwargs
            )

        else:
            if dtype is not None or len(args) == 5:
                raise RuntimeError(
                    "from_buffer: 'dtype' can only be specified in "
                    "UntypedStorage.from_buffer and TypedStorage.from_buffer"
                )
            if device is not None:
                raise RuntimeError(
                    "from_buffer: 'device' can only be specified in "
                    "UntypedStorage.from_buffer and TypedStorage.from_buffer"
                )

            dtype = cls._dtype
            untyped_storage = torch.UntypedStorage.from_buffer(
                *args, dtype=dtype, **kwargs
            )

        return TypedStorage(wrap_storage=untyped_storage, dtype=dtype, _internal=True)

    def _to(self, dtype):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Argument 'dtype' must be torch.dtype, not {type(dtype)}")
        storage = (
            torch.tensor([], dtype=self.dtype, device=self.device)
            .set_(self)
            .to(dtype)
            ._typed_storage()
        )
        if storage.data_ptr() == self.data_ptr():
            storage = storage.clone()
        return storage

    def double(self):
        """Casts this storage to double type."""
        _warn_typed_storage_removal()
        return self._to(torch.double)

    def float(self):
        """Casts this storage to float type."""
        _warn_typed_storage_removal()
        return self._to(torch.float)

    def half(self):
        """Casts this storage to half type."""
        _warn_typed_storage_removal()
        return self._to(torch.half)

    def long(self):
        """Casts this storage to long type."""
        _warn_typed_storage_removal()
        return self._to(torch.long)

    def int(self):
        """Casts this storage to int type."""
        _warn_typed_storage_removal()
        return self._to(torch.int)

    def short(self):
        """Casts this storage to short type."""
        _warn_typed_storage_removal()
        return self._to(torch.short)

    def char(self):
        """Casts this storage to char type."""
        _warn_typed_storage_removal()
        return self._to(torch.int8)

    def byte(self):
        """Casts this storage to byte type."""
        _warn_typed_storage_removal()
        return self._to(torch.uint8)

    def bool(self):
        """Casts this storage to bool type."""
        _warn_typed_storage_removal()
        return self._to(torch.bool)

    def bfloat16(self):
        """Casts this storage to bfloat16 type."""
        _warn_typed_storage_removal()
        return self._to(torch.bfloat16)

    def complex_double(self):
        """Casts this storage to complex double type."""
        _warn_typed_storage_removal()
        return self._to(torch.cdouble)

    def complex_float(self):
        """Casts this storage to complex float type."""
        _warn_typed_storage_removal()
        return self._to(torch.cfloat)

    def float8_e5m2(self):
        """Casts this storage to float8_e5m2 type"""
        _warn_typed_storage_removal()
        return self._to(torch.float8_e5m2)

    def float8_e4m3fn(self):
        """Casts this storage to float8_e4m3fn type"""
        _warn_typed_storage_removal()
        return self._to(torch.float8_e4m3fn)

    def float8_e5m2fnuz(self):
        """Casts this storage to float8_e5m2fnuz type"""
        _warn_typed_storage_removal()
        return self._to(torch.float8_e5m2fnuz)

    def float8_e4m3fnuz(self):
        """Casts this storage to float8_e4m3fnuz type"""
        _warn_typed_storage_removal()
        return self._to(torch.float8_e4m3fnuz)

    @classmethod
    def from_file(cls, filename, shared, size):
        """from_file(filename, shared=False, size=0) -> Storage

        Creates a CPU storage backed by a memory-mapped file.

        If ``shared`` is ``True``, then memory is shared between all processes.
        All changes are written to the file. If ``shared`` is ``False``, then the changes on
        the storage do not affect the file.

        ``size`` is the number of elements in the storage. If ``shared`` is ``False``,
        then the file must contain at least ``size * sizeof(Type)`` bytes
        (``Type`` is the type of storage). If ``shared`` is ``True`` the file will be created if needed.

        Args:
            filename (str): file name to map
            shared (bool): whether to share memory (whether ``MAP_SHARED`` or ``MAP_PRIVATE`` is passed to the
                            underlying `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_)
            size (int): number of elements in the storage
        """
        _warn_typed_storage_removal()
        if cls == TypedStorage:
            raise RuntimeError("from_file can only be called on derived classes")
        untyped_storage = UntypedStorage.from_file(
            filename, shared, size * torch._utils._element_size(cls.dtype)
        )
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
        (
            manager_handle,
            storage_handle,
            size,
        ) = self._untyped_storage._share_filename_cpu_(*args, **kwargs)
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

        if self.device.type not in [
            "cpu",
            "cuda",
            "hpu",
            torch._C._get_privateuse1_backend_name(),
        ]:
            return None

        module = (
            torch if self.device.type == "cpu" else getattr(torch, self.device.type)
        )

        try:
            return getattr(module, storage_name)
        except AttributeError:
            return None


TypedStorage.type.__doc__ = _type.__doc__
TypedStorage.cuda.__doc__ = _StorageBase.cuda.__doc__
TypedStorage.hpu.__doc__ = _StorageBase.hpu.__doc__
TypedStorage.to.__doc__ = _to.__doc__


class _LegacyStorageMeta(type):
    dtype: torch.dtype

    def __instancecheck__(cls, instance):
        if type(instance) == TypedStorage:
            cls_device = _get_device_from_module(cls.__module__)
            return (cls_device == instance.device.type) and (
                cls.dtype == instance.dtype
            )
        return False


class _LegacyStorage(TypedStorage, metaclass=_LegacyStorageMeta):
    @classmethod
    def _new_shared(cls, size):
        """Create a new storage in shared memory with the same data type."""
        untyped_storage = torch.UntypedStorage._new_shared(size * cls()._element_size())
        return cls(wrap_storage=untyped_storage)

    @classmethod
    def _release_ipc_counter(cls, *args, **kwargs):
        return torch.UntypedStorage._release_ipc_counter_cuda(*args, **kwargs)

    @classmethod
    def _new_shared_filename(cls, manager, obj, size):
        bytes_size = size * torch._utils._element_size(cls.dtype)
        return cls(
            wrap_storage=torch.UntypedStorage._new_shared_filename_cpu(
                manager, obj, bytes_size
            )
        )


def _get_dtype_from_pickle_storage_type(pickle_storage_type: str):
    try:
        return _storage_type_to_dtype_map()[pickle_storage_type]
    except KeyError as e:
        raise KeyError(
            f'pickle storage type "{pickle_storage_type}" is not recognized'
        ) from e
