# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from enum import Enum, auto
from functools import reduce
import io
import os
import pickle
from types import TracebackType
from typing import IO, Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch.utils._pytree import tree_map
from torch.serialization import DEFAULT_PROTOCOL as DEFAULT_PROTOCOL

DEFAULT_CHUNK_SIZE = 2048 * 2048


class TensorSerialization:
    @staticmethod
    def _get_num_chunks(input_tensor: torch.Tensor, chunk_size_bytes: int = DEFAULT_CHUNK_SIZE) -> int:
        """Returns the number of chunks that the given tensor can be divided into."""
        size_in_bytes = input_tensor.nelement() * input_tensor.element_size()
        num_chunks = (size_in_bytes + (chunk_size_bytes - 1)) // chunk_size_bytes
        return num_chunks

    @staticmethod
    def _tensor_to_bytes_chunks(
        input_tensor: torch.Tensor, chunk_idx: int, chunk_size_bytes: int = DEFAULT_CHUNK_SIZE
    ) -> bytes:
        """Called multiple (_get_num_chunks(input_tensor) times, each call converts provided tensor chunk into a byte array containing chunk_size_bytes."""
        size_in_bytes = input_tensor.nelement() * input_tensor.element_size()
        assert chunk_idx < TensorSerialization._get_num_chunks(input_tensor, chunk_size_bytes)
        input_tensor_np = input_tensor.detach().numpy().view(np.uint8).reshape(-1)
        chunk_start = chunk_idx * chunk_size_bytes
        chunk_end = min(size_in_bytes, chunk_start + chunk_size_bytes)
        return input_tensor_np[chunk_start:chunk_end].tobytes()


    @staticmethod
    def write(input_tensor: torch.Tensor, filename: str, file_offset_bytes: int = 0) -> None:
        """Populates the file with the data stored in the given tensor."""
        num_chunks = TensorSerialization._get_num_chunks(input_tensor)
        file_flags = "r+b" if os.path.exists(filename) else "wb"
        with open(filename, file_flags) as f:
            f.seek(file_offset_bytes)
            for i in range(num_chunks):
                f.write(TensorSerialization._tensor_to_bytes_chunks(input_tensor, i))


    @staticmethod
    def read(input_tensor: torch.Tensor, filename: str, file_offset_bytes: int = 0) -> None:
        """Populates the given tensor with the data stored in a file."""
        size_in_bytes = input_tensor.nelement() * input_tensor.element_size()
        chunk_size_bytes = DEFAULT_CHUNK_SIZE
        num_chunks = TensorSerialization._get_num_chunks(input_tensor)
        input_tensor_np = input_tensor.detach().numpy()
        input_tensor_mv = memoryview(input_tensor_np.view(dtype=np.uint8).reshape(-1))
        with io.open(filename, "rb") as f:
            f.seek(file_offset_bytes)
            for i in range(num_chunks):
                chunk_start = i * chunk_size_bytes
                chunk_end = min(size_in_bytes, chunk_start + chunk_size_bytes)
                data_read = f.readinto(input_tensor_mv[chunk_start:chunk_end])
                if data_read != chunk_end - chunk_start:
                    raise RuntimeError(
                        f"Attempted to read {chunk_end - chunk_start} more bytes from {filename}, but only read: {data_read} bytes. Total Bytes read = {chunk_start + data_read}, total bytes expected: {size_in_bytes}"
                    )


class StorageState(Enum):
    """
    Simple enum to indicate whether the tensor handle is pointing
    to data on disk or memory. This is useful for asserting on
    whether the tensor is available for operations (ON_CPU_CLEAN or ON_CPU_DIRTY)
    or if it needs to be moved from disk to CPU (ON_DISK)  or device,
    or if the tensor is available and has diverged from the copy on disk
    (ON_CPU_DIRTY)
    """

    UNALLOCATED = auto()
    ON_DISK = auto()
    ON_CPU_CLEAN = auto()
    ON_CPU_DIRTY = auto()


class SsdTensorHandle(torch.Tensor):
    """
    This class extends from torch.Tensor and represents a Tensor which is backed by SSD storage.
    The SsdTensorHandle object can point to a file or a tensor and there are corresponding functions to read
    data into the tensor that is an attribute of the SsdTensorHandle object or write the tensor to file. At any
    point in time the Tensor may be in memory or on disk.

    Class Variables:
        override_directory_path: This variable is used by CheckpointPathContextManager to modify the path to any
            SsdTensorHandles that are saved to a checkpoint via pickling (e.g. torch.save)

    Args:
        shape torch.Size: Shape of the tensor that is represented by the handle.
        dtype: torch.dtype: Dtype of the tensor that is represented by the handle.
        requires_grad: bool: Property of the tensor that is represeneted by the handle.
        device: torch.device: determines on which device the data is stored (normally cpu
                or gpu (cuda))
        flush_on_dirty: bool: If true, anytime mark_dirty() is called, it will immediately
                trigger a write to disk. Otherwise, storage_state will transition to ON_CPU_DIRTY
        allow_unsafe_changes: bool: permit certain unsafe practices when overwriting the SsdTensorHandle's
                .data attribute. Examples of unsafe practices are: changing when storage_state is
                ON_CPU_DIRTY, altering tensor shape, or altering requires_grad value.

    Returns:
        A SsdTensorHandle object representing a Tensor.
    """

    override_directory_path: Optional[str] = None

    @staticmethod
    def __new__(
        cls: Type[SsdTensorHandle],
        shape: torch.Size,
        dtype: torch.dtype,
        requires_grad: bool = False,
        device: torch.device = torch.device("cpu"),
        flush_on_dirty: bool = True,
        allow_unsafe_changes: bool = False,
    ) -> SsdTensorHandle:
        r = super(SsdTensorHandle, cls)._make_wrapper_subclass(cls, shape, dtype=dtype, requires_grad=requires_grad, device=device)  # type: ignore
        return r

    def __init__(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        requires_grad: bool,
        device: torch.device = torch.device("cpu"),
        flush_on_dirty: bool = True,
        allow_unsafe_changes: bool = False,
    ) -> None:
        self._unpickle_f: Optional[Union[BinaryIO, IO[bytes]]] = None

        self._shape = shape
        if len(shape) == 0:
            self._numel = 0
        else:
            self._numel = reduce((lambda x, y: x * y), shape)
        self._dtype = dtype
        # valid if offloaded to file
        self.filename = ""
        self.offset = -1
        # valid if loaded to memory
        self.tensor: Optional[torch.Tensor] = None
        self.storage_state = StorageState.UNALLOCATED
        self.flush_on_dirty = flush_on_dirty
        self.allow_unsafe_changes = allow_unsafe_changes

    def mark_dirty(self) -> None:
        assert self.tensor is not None
        assert self.storage_state in [StorageState.ON_CPU_CLEAN, StorageState.ON_CPU_DIRTY]
        self.storage_state = StorageState.ON_CPU_DIRTY
        # hack to force write on mark_dirty
        if self.flush_on_dirty:
            self.to_file()

    @classmethod
    def from_file(
        cls, shape: torch.Size, dtype: torch.dtype, filename: str, offset: int = 0, requires_grad: bool = False
    ) -> SsdTensorHandle:
        """Returns a new SsdTensorHandle from a file."""
        handle = cls(shape=shape, dtype=dtype, requires_grad=requires_grad)
        handle.point_to_file(filename, offset=offset)
        return handle

    @classmethod
    def from_tensor(cls: Type[SsdTensorHandle], tensor: torch.Tensor) -> SsdTensorHandle:
        """Returns a new SsdTensorHandle from a tensor. Most common use-case."""
        handle = cls(shape=tensor.shape, dtype=tensor.dtype, requires_grad=tensor.requires_grad, device=tensor.device)
        handle.point_to_tensor(tensor)
        return handle

    def is_available(self) -> bool:
        return self.tensor is not None

    def get_tensor(self) -> torch.Tensor:
        assert self.tensor is not None
        return self.tensor

    def set_file_params(self, filename: str, offset: int) -> None:
        self.filename = filename
        self.offset = offset

    def point_to_file(self, filename: str, offset: int) -> None:
        self.set_file_params(filename, offset)
        self.tensor = None
        self.storage_state = StorageState.ON_DISK

    def point_to_tensor(self, tensor: torch.Tensor) -> None:
        assert self.tensor is None
        if not self.allow_unsafe_changes:
            assert self._shape == tensor.shape
        assert self._dtype == tensor.dtype
        self.tensor = tensor
        self.storage_state = StorageState.ON_CPU_DIRTY

    # if resizing a handle that is part of an ssd buffer, care must be taken that the new size
    # doesn't conflict with adjacent handles!
    def point_to_resized_tensor(self, tensor: torch.Tensor) -> None:
        assert self._dtype == tensor.dtype
        self._shape = tensor.shape
        self.tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        """Returns the tensor represented by the SsdTensorHandle object.

        If the tensor is on disk, it is copied into the .tensor attribute and returned.
        """
        if self.tensor is not None:
            return self.tensor
        else:
            if self.device != torch.device("cpu"):
                raise RuntimeError(
                    f"to_tensor called on an SsdTensorHandle when the tensor has been offloaded to disk. self.device = {self.device}, it should be {torch.device('cpu')}. Some unexpected .data override has occured!!"
                )
            result_tensor = torch.empty(size=self.shape, dtype=self.dtype, requires_grad=self.requires_grad, device=self.device)
            self.copy_into_tensor(result_tensor)
            self.tensor = result_tensor
            self.storage_state = StorageState.ON_CPU_CLEAN
            return self.tensor

    def to_file(self, permit_when_tensor_none: bool = False, release_tensor_after_write: bool = True) -> None:
        """Saves the tensor to disk and releases memory if specified."""
        assert self.tensor is not None or permit_when_tensor_none

        # if it's available in Memory but not modified, no need to write-back
        if self.tensor is not None:
            if self.storage_state is StorageState.ON_CPU_DIRTY:
                if self.device != torch.device("cpu"):
                    raise RuntimeError(
                        f"to_file called on an SsdTensorHandle when self.device = {self.device}, it should be {torch.device('cpu')}. Some unexpected .data override has occured!!"
                    )
                TensorSerialization.write(self.tensor, self.filename, self.offset * self.tensor.element_size())
            if release_tensor_after_write:
                self.tensor = None
                self.storage_state = StorageState.ON_DISK
            else:
                self.storage_state = StorageState.ON_CPU_CLEAN

    def copy_into_tensor(self, tensor: torch.Tensor) -> None:
        """Copies SsdTensorHandle's data into the given tensor.

        If the tensor is in memory, this function copies the data
        into the passed in tensor. Otherwise, it reads from file into tensor,
        using the read() function.
        This does not modify modify self.tensor unlike the to_tensor()
        function. This can be useful for calls like named_parameters() when
        the tensor is already offloaded to disk.
        """
        # ideally this should be checked but .data shenanigans forces it to
        # be disabled due to the way FSDP shards parameters
        # assert self._shape == tensor.shape
        assert self._dtype == tensor.dtype
        if self.tensor is not None:
            tensor.copy_(self.tensor)
        else:
            TensorSerialization.read(tensor, self.filename, self.offset * tensor.element_size())

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore
        """Intercepts all operations performed on this handle object.

        Before any operation, the tensor attribute is unwrapped from the handle
        and used in the operation. If we detect changes to the tensor by analyzing
        the function name and whether kwargs["out"] specifies a SsdTensorHandle,
        we mark the tensor as dirty (which may trigger a write to the file, depending
        on the .flush_on_dirty attribute.
        """
        func_name = func.overloadpacket.__name__
        ssd_tensor_handles = []

        def unwrap(e: Any) -> torch.Tensor:
            if isinstance(e, SsdTensorHandle):
                t = e.to_tensor()
                ssd_tensor_handles.append(e)
                return t
            else:
                return e

        r = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        for e in ssd_tensor_handles:
            inplace_is_this_tensor = (
                (func_name.endswith("_") and not func_name.endswith("__")) or func_name.startswith("__i")
            ) and e is args[0]
            out_is_this_tensor = False if "out" not in kwargs else e is kwargs["out"]
            if inplace_is_this_tensor or out_is_this_tensor:
                e.mark_dirty()
        return r

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "data":
            assert isinstance(value, torch.Tensor)
            if not self.allow_unsafe_changes:
                # Respect .data changes, and the user better know what they are doing!
                if self.storage_state == StorageState.ON_CPU_DIRTY:
                    raise RuntimeError(
                        "Attempting to override tensor when the existing tensor is dirty, this is an error!"
                    )
                if value.shape != self.shape:
                    raise RuntimeError(
                        f"Attempting to override tensor metadata using .data to change shape of tensor. Orig shape: {self.shape} New shape: {value.shape}"
                    )
                if value.requires_grad != self.requires_grad:
                    raise RuntimeError(
                        f"Attempting to override tensor metadata using .data to change requires_grad. Orig value: {self.requires_grad} New value: {value.requires_grad}"
                    )
            self.tensor = value
        super(SsdTensorHandle, self).__setattr__(name, value)

    @classmethod
    def __unpickle__(
        cls: Type[SsdTensorHandle], shape: torch.Size, dtype: torch.dtype, requires_grad: bool, filename: str
    ) -> SsdTensorHandle:
        result = cls(shape, dtype, requires_grad)
        result.point_to_file(filename, 0)
        result._unpickle_f = io.open(result.filename, "wb")
        return result

    def __del__(self) -> None:
        if getattr(self, "_unpickle_f", None):
            self._unpickle_f.close()

    def __reduce_ex__(self, proto: int) -> Tuple[Any, Any, Any, Any]:
        byte_iter = None
        filename = self.filename
        if self.override_directory_path is not None:
            head, tail = os.path.split(self.filename)
            filename = os.path.join(self.override_directory_path, tail)
        if self.is_available():
            byte_iter = iter(TensorChunkingIterator(self.tensor))  # ignore: type
        else:
            byte_iter = iter(
                FileChunkingIterator(self.filename, expected_size_bytes=self.numel() * self.element_size())
            )
        return (
            self.__unpickle__,  # Callable
            # Args to the callable above
            (self._shape, self._dtype, self.requires_grad, filename),
            None,
            byte_iter,
        )

    def append(self, item: bytes) -> None:
        assert self._unpickle_f
        self._unpickle_f.write(item)

    def extend(self, items: List[bytes]) -> None:
        for i in items:
            self.append(i)


class CheckpointPathContextManager:
    """
    This Context allows the user to override the directory path when pickling an SsdTensorHandle Object.
    It is needed because the filename which the SsdTensorHandle points to (and is used when unpickling)
    is already baked into the pickled data.

    Consider the following example code
        ssd_handle = SsdTensorHandle.from_tensor(ref_tensor)
        ssd_handle.set_file_params('/home/user/handle.bin', 0)
        torch.save(ssd_handle, '/home/user/checkpoint.pkl')
        ssd_handle += 1
        ssd_handle.to_file()
        ssd_handle2 = torch.load('/home/user/checkpoint.pkl')

        print(f"handles are equal: {torch.equals(ssd_handle, ssd_handle2)}")

    One would expect this to print False, however unintuitively it will print True.
    ssd_handle.filename and ssd_handle2.filename are equal. This means that
    when we execute torch.load, we read from the .pkl file and write the result into
    /home/user/handle.bin, clobbering the updated result from `ssd_handle += 1`

    We want to give the user the possibility of not clobbering the data using this
    Context Manager.

        ssd_handle = SsdTensorHandle.from_tensor(ref_tensor)
        ssd_handle.set_file_params('/home/user/handle.bin', 0)
        with CheckpointPathContextManager(override_path='/home/user/checkpoint_data/'):
            torch.save(ssd_handle, '/home/user/checkpoint.pkl')
        ssd_handle += 1
        ssd_handle.to_file()
        ssd_handle2 = torch.load('/home/user/checkpoint.pkl')

        print(f"handles are equal: {torch.equals(ssd_handle, ssd_handle2)}")

    This code results with ssd_handle.filename = '/home/user/handle.bin' and ssd_handle2.filename =
    `/home/user/checkpoint_data/handle.bin'. Therefore the torch.load won't clobber ssd_handle, and
    the printed result is False.

    """

    def __init__(self, override_path: str) -> None:
        self.old_path = SsdTensorHandle.override_directory_path
        self.override_path = override_path

    def __enter__(self) -> None:
        SsdTensorHandle.override_directory_path = self.override_path

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exec_traceback: Optional[TracebackType],
    ) -> None:
        SsdTensorHandle.override_directory_path = self.old_path


class SsdParameter(SsdTensorHandle, torch.nn.Parameter):
    @classmethod
    def from_tensor(cls: Type[SsdParameter], tensor: SsdTensorHandle) -> SsdParameter:  # type: ignore
        r = cls(tensor.shape, tensor.dtype, tensor.requires_grad, device=tensor.device)
        r.point_to_tensor(tensor)
        return r

    @staticmethod
    def __new__(
        cls: Type[SsdParameter],
        shape: torch.Size,
        dtype: torch.dtype,
        requires_grad: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> SsdParameter:
        r = super(SsdParameter, cls).__new__(cls, shape=shape, dtype=dtype, requires_grad=requires_grad, device=device)
        return r  # type: ignore

    def __init__(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        requires_grad: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super(SsdParameter, self).__init__(shape=shape, dtype=dtype, requires_grad=requires_grad, device=device)


class SsdFlatParameter(SsdParameter):
    """A parameter that is initialized from a list of parameters and can be
    turned into a list of views as needed.

    This class should eventually be moved to fairscale/nn/misc/flatten_params_wrapper.py
    """

    def __new__(
        cls: Type[SsdFlatParameter],
        shapes: Sequence[torch.Size],
        dtype: torch.dtype,
        requires_grad: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> SsdFlatParameter:
        """Make an object using the parent's __new__ function."""

        # A empty of non-list input doesn't make sense.
        if not isinstance(shapes, (list, tuple)) or len(shapes) == 0:
            raise ValueError("An non-empty list or tuple argument is needed")

        size = sum([np.prod(s) for s in shapes])
        r = super(SsdFlatParameter, cls).__new__(
            cls, torch.Size((size,)), dtype=dtype, requires_grad=requires_grad, device=device
        )
        return r  # type: ignore

    def __init__(
        self,
        shapes: Sequence[torch.Size],
        dtype: torch.dtype,
        requires_grad: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize the _param_numels and _param_shapes lists."""
        self._param_shapes = shapes
        self._param_numels = [np.prod(s) for s in shapes]
        total_numels = sum(self._param_numels)
        assert (
            self.numel() <= total_numels
        ), f"Something wrong with __new__ method, {self.numel()} vs. {sum(self._param_numels)}"

        self.views: List[SsdFlatParameterView] = []
        # These are set by FPW class below, not by this class itself.
        self._param_infos: List[Tuple[str, torch.nn.Module, str]] = []
        self._shared_param_infos: List[Tuple[str, str, torch.nn.Module, str, torch.nn.Module, str]] = []

        super(SsdFlatParameter, self).__init__(
            shape=torch.Size((total_numels,)), dtype=dtype, requires_grad=requires_grad
        )

    def __setattr__(self, name: str, value: Any) -> None:
        super(SsdFlatParameter, self).__setattr__(name, value)
        if name == "data":
            # if .data has changed, we need to totally destroy any existing views because things
            # like device might have changed. It won't destroy any pointers to those views outside
            # of here, however resetting self.views will trigger the old view's assertion in
            # __torch_dispatch__ that it is the current view of it's parent object
            self.views = []
            self._refresh_views()

    def _invalidate_views(self) -> None:
        for v in self.views:
            v.tensor = None

    @torch.enable_grad()
    def _refresh_views(self) -> None:
        if self._shape != self.shape:
            self.views = []
            return
        if len(self.views) == 0:
            self.views = [s.view(v) for s, v in zip(self.split(self._param_numels), self._param_shapes)]  # type: ignore
        else:
            for v, t, s in zip(self.views, self.tensor.split(self._param_numels), self._param_shapes):
                v.tensor = t.view(s)

    def get_param_views(self, external_data: Optional[torch.Tensor] = None) -> Iterator[torch.Tensor]:
        """Return a generator of views that map to the original parameters."""
        # Note, self.data could be sharded, so its numel is <= to the sum.
        """
        assert self.data.numel() <= sum(
            self._param_numels
        ), f"Incorrect internal state {self.data.numel()} vs. {sum(self._param_numels)}"
        """
        if external_data is not None:
            if external_data.numel() != sum(self._param_numels):
                raise ValueError(
                    f"Incorrect numel of supplied data: got {external_data.numel()} but expected {sum(self._param_numels)}"
                )
            return (t.view(s) for (t, s) in zip(external_data.split(self._param_numels), self._param_shapes))
        else:
            # this needs to return SsdFlatParameterViews
            if not self.is_available():
                self.to_tensor()

            if len(self.views) == 0:
                raise RuntimeError(
                    "Trying to call get_param_views when self.views is empty, this means that .data games have been played and the current .data shape doesn't match the constructed shape."
                )
            return (v for v in self.views)

    def metadata(self) -> Tuple[List[str], Sequence[torch.Size], List[int]]:
        """Return tuple of (names, shapes, numels) metadata for this flat parameter."""
        names = [".".join([m, n]) if m else n for (m, _, n) in self._param_infos]
        return names, self._param_shapes, self._param_numels

    @classmethod
    def from_tensors(
        cls: Type[SsdFlatParameter],
        tensors: Sequence[torch.Tensor],
        direct_to_file: bool = False,
        filename: str = "",
        offset: int = 0,
    ) -> "SsdFlatParameter":
        """Returns a new SsdFlatParameter from a sequence of tensors."""
        assert (
            len(tensors) > 0
        ), "SsdFlatParameter.from_tensors must be called with at least one tensor in the tensors argument"

        # Flattening involves (1) making a tensor flat (i.e. single dimensional) and (2) making a module
        # heirarchy flat (using a single tensor to replace a tree of tensors). Therefore,
        # adding back nesting and heirarchy is counter-productive. If nesting is encountered
        # in the future, the reasonable thing to do is likely for the top level SsdFlatParameter to
        # absorb the nested one and keep the result flat, free from hierarchy.
        if any(isinstance(t, SsdFlatParameter) for t in tensors):
            raise ValueError("Nesting SsdFlatParameter is not supported")

        requires_grad = tensors[0].requires_grad
        dtype = tensors[0].dtype
        device = tensors[0].device
        for t in tensors:
            if t.requires_grad != requires_grad:
                raise RuntimeError("Not all tensors have identical requires_grad option")
            if t.dtype != dtype:
                raise RuntimeError("Not all tensors have identical dtype option")
            if t.device != device:
                raise RuntimeError("Not all tensors have identical device option")
        handle = cls(
            shapes=[t.size() for t in tensors],
            dtype=tensors[0].dtype,
            requires_grad=tensors[0].requires_grad,
            device=device,
        )
        handle.set_file_params(filename, offset)
        if direct_to_file:
            assert filename != ""
            offset = offset
            for t in tensors:
                TensorSerialization.write(t, handle.filename, offset)
                offset += t.numel() * t.element_size()

            handle.storage_state = StorageState.ON_DISK
        else:
            tensor = torch.cat(
                [t.reshape(-1) if isinstance(t, torch.nn.Parameter) else t.reshape(-1) for t in tensors],
                0,
            ).detach()
            tensor.requires_grad_()
            handle.point_to_tensor(tensor)
        return handle

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore
        func_name = func.overloadpacket.__name__
        r = super(SsdFlatParameter, cls).__torch_dispatch__(func, types, args, kwargs)  # type: ignore
        if func_name.startswith("split"):
            assert isinstance(args[0], SsdFlatParameter)
            parent = args[0]
            return [SsdFlatParameterView(parent, t, idx) for idx, t in enumerate(r)]
        else:
            return r

    # need to subclass these methods to support Views
    def point_to_tensor(self, tensor: torch.Tensor) -> None:
        super(SsdFlatParameter, self).point_to_tensor(tensor)
        self._refresh_views()

    def point_to_file(self, filename: str, offset: int) -> None:
        super(SsdFlatParameter, self).point_to_file(filename, offset)
        self._invalidate_views()

    def to_tensor(self) -> torch.Tensor:
        call_refresh_views = False
        if self.tensor is None:
            call_refresh_views = True
        result = super(SsdFlatParameter, self).to_tensor()
        if call_refresh_views:
            self._refresh_views()
        return result

    def to_file(self, permit_when_tensor_none: bool = False, release_tensor_after_write: bool = True) -> None:
        super(SsdFlatParameter, self).to_file(permit_when_tensor_none, release_tensor_after_write)
        self._invalidate_views()

    @classmethod
    def __unpickle_SFP__(
        cls: Type[SsdFlatParameter],
        shapes: Sequence[torch.Size],
        dtype: torch.dtype,
        requires_grad: bool,
        filename: str,
    ) -> SsdFlatParameter:
        result = cls(shapes, dtype, requires_grad)
        result.point_to_file(filename, 0)
        result._unpickle_f = io.open(result.filename, "wb")
        return result

    def __reduce_ex__(self, proto: int) -> Tuple[Any, Any, Any, Any]:
        byte_iter = None
        filename = self.filename
        if self.override_directory_path is not None:
            head, tail = os.path.split(self.filename)
            filename = os.path.join(self.override_directory_path, tail)
        if self.is_available():
            byte_iter = iter(TensorChunkingIterator(self.tensor))
        else:
            byte_iter = iter(
                FileChunkingIterator(self.filename, expected_size_bytes=self.numel() * self.element_size())
            )
        return (
            self.__unpickle_SFP__,  # Callable
            # Args to the callable above
            (self._param_shapes, self._dtype, self.requires_grad, filename),
            None,
            byte_iter,
        )


class SsdFlatParameterView(torch.Tensor):
    """
    Represents a view into an SsdFlatParameter. It is needed due to FSDP's usage of flattening parameters.
    """

    def __new__(
        cls: Type[SsdFlatParameterView], parent: SsdFlatParameter, tensor: torch.Tensor, id: int
    ) -> SsdFlatParameterView:
        r = super(SsdFlatParameterView, cls)._make_wrapper_subclass(cls, tensor.shape, dtype=tensor.dtype, requires_grad=tensor.requires_grad, device=tensor.device)  # type: ignore
        return r

    def __init__(self: SsdFlatParameterView, parent: SsdFlatParameter, tensor: torch.Tensor, id: int) -> None:
        self.parent = parent
        self.tensor: Optional[torch.Tensor] = tensor
        self.id = id

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore
        """Intercepts all operations performed on this handle object.

        Before any operation, the tensor attribute is unwrapped from the handle
        and used in the operation. We maintain a refernce to the tensor and its current
        versions to track if modifications have been made. If we detect changes to the
        tensor, we write it to the file maintained by the Handle.
        """
        func_name = func.overloadpacket.__name__
        ssd_tensor_handles = []

        def unwrap(e: Any) -> torch.Tensor:
            if isinstance(e, SsdFlatParameterView):
                if not e.parent.is_available():
                    e.parent.to_tensor()
                # first condition is to take care of the case where we are first constructing e.parent.views as a list comprehension which hasn't
                # completed yet
                if len(e.parent.views) != 0 and e is not e.parent.views[e.id]:
                    raise RuntimeError(
                        "This view should no longer be used as the parent object has had it's .data overwritten (e.parent.views[e.id])!!!"
                    )
                # e.parent will ensure that e.tensor is valid and points to tensor view
                t = e.tensor
                ssd_tensor_handles.append(e)
                return t
            else:
                return e

        r = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        for e in ssd_tensor_handles:
            inplace_is_this_tensor = (
                (func_name.endswith("_") and not func_name.endswith("__")) or func_name.startswith("__i")
            ) and e is args[0]
            out_is_this_tensor = False if "out" not in kwargs else e is kwargs["out"]
            if inplace_is_this_tensor or out_is_this_tensor:
                e.parent.mark_dirty()

        if func_name.startswith("view"):
            assert isinstance(args[0], SsdFlatParameterView)
            flat_view = args[0]
            return SsdFlatParameterView(flat_view.parent, r, flat_view.id)
        return r


class PropertizeModule():
    """This code is taken mostly from pytorch core parameterization
       pytorch/torch/nn/utils/parametrize.py. It's purpose is to
       turn a module's registered nn.Parameter (which is normally an
       attribute) into a python property. This means that we supply
       a .getter and .setter function so we can override them if necessary.
    """

    @staticmethod
    def __inject_new_class(module: torch.nn.Module) -> None:
        r"""Sets up a module to be parametrized.

        This works by substituting the class of the module by a class
        that extends it to be able to inject a property

        Args:
            module (nn.Module): module into which to inject the property
        """
        cls = module.__class__

        def getstate(self):  # type: ignore
            raise RuntimeError(
                "Serialization of parametrized modules is only "
                "supported through state_dict(). See:\n"
                "https://pytorch.org/tutorials/beginner/saving_loading_models.html"
                "#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training"
            )

        param_cls = type(
            f"Parametrized{cls.__name__}",
            (cls,),
            {
                "__getstate__": getstate,
            },
        )

        module.__class__ = param_cls
        module.__override_properties__: Dict[str, Callable[[], torch.Tensor]] = {}  # type: ignore


    @staticmethod
    def __inject_property(module: torch.nn.Module, property_name: str) -> None:
        r"""Injects a property into module[property_name].

        It assumes that the class in the module has already been modified from its
        original one using __inject_new_class and that the tensor under :attr:`property_name`
        has already been moved out

        Args:
            module (nn.Module): module into which to inject the property
            property_name (str): name of the name of the property to create
        """

        def get_propertized(self: torch.nn.Module) -> torch.Tensor:
            prop: Callable[[], torch.Tensor] = self.__override_properties__[property_name]  # type: ignore
            # If caching is not active, this function just evaluates the propertization
            return prop()

        def set_original(self: torch.nn.Module, value: Callable[[], torch.Tensor]) -> None:
            self.__override_properties__[property_name] = value  # type: ignore

        def del_fn(self: torch.nn.Module) -> None:
            _remove_property(self, property_name)

        setattr(module.__class__, property_name, property(get_propertized, set_original, del_fn))


    @staticmethod
    def register_property(module: torch.nn.Module, property_name: str, property_value: Callable[[], torch.Tensor]) -> None:
        has_injected_class = hasattr(module, "__override_properties__")
        if not has_injected_class:
            PropertizeModule.__inject_new_class(module)
            if hasattr(module, property_name):
                delattr(module, property_name)
        module.__override_properties__[property_name] = property_value  # type: ignore
        PropertizeModule.__inject_property(module, property_name)


    @staticmethod
    def remove_property(module: torch.nn.Module, property_name: str, new_property_value: Optional[Any] = None) -> None:
        delattr(module.__class__, property_name)
        del module.__override_properties__[property_name]  # type: ignore

        # Roll back the parametrized class if no other buffer or parameter
        # is currently parametrized in this class
        if len(module.__override_properties__) == 0:  # type: ignore
            delattr(module, "__override_properties__")
            # Restore class
            orig_cls = module.__class__.__bases__[0]
            module.__class__ = orig_cls
        if new_property_value is not None:
            setattr(module.__class__, property_name, new_property_value)


class SsdFlatParameterViewProperty:
    """
    Allows for a mutable view to replace a layer's trainable parameters.
    This is needed since FSDP is changing tensor metadata (.data) under the covers,
    SsdFlatParameter cannot just rely on this since each view (of type SsdFlatParameterView) has
    an internal representation. So every time we access a view, we need to
    make sure we get the up-to-date(with the updated .data info) version, and not
     the original version.
    """

    def __init__(self, parent: SsdFlatParameter, view_id: int) -> None:
        super().__init__()
        self.parent = parent
        self.view_id = view_id

    def __call__(self) -> SsdFlatParameterView:
        return self.parent.views[self.view_id]

# Classes supporting torch.save/load
class TorchSaver:
    def __init__(self) -> None:
        self.pickle_module = DisableMemoizationPicklerModule

    def save(
        self, obj: Any, f: Union[str, os.PathLike, BinaryIO, IO[bytes]], pickle_protocol: int = DEFAULT_PROTOCOL
    ) -> None:
        torch.serialization.save(
            obj, f, self.pickle_module, pickle_protocol=pickle_protocol, _use_new_zipfile_serialization=False
        )


class DisableMemoizationPicklerModule:
    @staticmethod
    def Pickler(data_buf: io.BytesIO, protocol: int) -> pickle.Pickler:
        p = pickle.Pickler(data_buf, protocol)
        p.fast = True
        return p

    @staticmethod
    def dump(obj: Any, f: io.BytesIO, protocol: int) -> None:
        pickle.dump(obj, f, protocol)


class TensorChunkingIterator:
    """
    class is used when the SsdTensorHandle is already loaded into memory

    chunk_size_bytes determines how large each chunk that we break the tensor
    into. It is important to consider limiting the size because by when
    python unpickles an object, by default it will read up to 1000 list
    elements at a time. So memory usage while unpickling will be on the
    order of O(min(file_size, 1000 * chunk_size_bytes)).
    """

    def __init__(self, tensor: torch.Tensor, chunk_size_bytes: int = DEFAULT_CHUNK_SIZE) -> None:

        self.tensor = tensor
        self.chunk_size_bytes = chunk_size_bytes

    def __iter__(self) -> Iterator[bytes]:

        self.num_chunks = TensorSerialization._get_num_chunks(self.tensor, self.chunk_size_bytes)
        self.num_chunks_read = 0
        return self

    def __next__(self) -> bytes:
        if self.num_chunks_read >= self.num_chunks:
            raise StopIteration
        next_chunk = TensorSerialization._tensor_to_bytes_chunks(
            self.tensor, chunk_idx=self.num_chunks_read, chunk_size_bytes=self.chunk_size_bytes
        )

        self.num_chunks_read += 1

        return next_chunk


class FileChunkingIterator:
    """
    class is used when the SsdTensorHandle is only available on disk

    chunk_size_bytes determines how large each chunk that we break the file
    into. It is important to consider limiting the size because by when
    python unpickles an object, by default it will read up to 1000 list
    elements at a time. So memory usage while unpickling will be on the
    order of O(min(file_size, 1000 * chunk_size_bytes)).
    """

    def __init__(
        self, filename: str, expected_size_bytes: int = -1, chunk_size_bytes: int = DEFAULT_CHUNK_SIZE
    ) -> None:
        self.filename = filename
        self.file: Optional[Union[BinaryIO, IO[bytes]]] = None
        self.chunk_size_bytes = chunk_size_bytes
        self.expected_size_bytes = expected_size_bytes

    def __iter__(self) -> Iterator[bytes]:

        if self.expected_size_bytes != -1:
            file_size = os.stat(self.filename).st_size
            assert (
                file_size == self.expected_size_bytes
            ), f"FileChunkingIterator Failed, expecting file to be of size: {self.expected_size_bytes} but got {file_size}"
        self.file = io.open(self.filename, "rb", buffering=0)
        self.num_chunks_read = 0
        return self

    def __next__(self) -> bytes:
        assert self.file
        next_chunk = self.file.read(self.chunk_size_bytes)

        if len(next_chunk) == 0:
            raise StopIteration
        self.num_chunks_read += 1

        return next_chunk

    def __del__(self) -> None:
        self.file.close()


torch_saver = TorchSaver()
