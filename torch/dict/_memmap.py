from __future__ import annotations

import mmap
import os

import sys
import tempfile
from multiprocessing import util
from multiprocessing.context import reduction
from pathlib import Path
from typing import Any, overload

import numpy as np

import torch

from torch import distributed as dist

from torch.multiprocessing.reductions import ForkingPickler

from .utils import implement_for

try:
    if dist.is_available():
        from torch.distributed._tensor.api import DTensor
    else:
        raise ImportError
except ImportError:

    class DTensor(torch.Tensor):  # noqa: D101
        ...


class MemoryMappedTensor(torch.Tensor):
    """A Memory-mapped Tensor.

    Supports filenames or file handlers.

    The main advantage of MemoryMappedTensor resides in its serialization methods,
    which ensure that the tensor is passed through queues or RPC remote calls without
    any copy.

    .. note::
      When used within RPC settings, the filepath should be accessible to both nodes.
      If it isn't the behaviour of passing a MemoryMappedTensor from one worker
      to another is undefined.

    MemoryMappedTensor supports multiple construction methods.

    Examples:
          >>> # from an existing tensor
          >>> tensor = torch.randn(3)
          >>> with tempfile.NamedTemporaryFile() as file:
          ...     memmap_tensor = MemoryMappedTensor.from_tensor(tensor, filename=file.name)
          ...     assert memmap_tensor.filename is not None
          >>> # if no filename is passed, a handler is used
          >>> tensor = torch.randn(3)
          >>> memmap_tensor = MemoryMappedTensor.from_tensor(tensor, filename=file.name)
          >>> assert memmap_tensor.filename is None
          >>> # one can create an empty tensor too
          >>> with tempfile.NamedTemporaryFile() as file:
          ...     memmap_tensor_empty = MemoryMappedTensor.empty_like(tensor, filename=file.name)
          >>> with tempfile.NamedTemporaryFile() as file:
          ...     memmap_tensor_zero = MemoryMappedTensor.zeros_like(tensor, filename=file.name)
          >>> with tempfile.NamedTemporaryFile() as file:
          ...     memmap_tensor = MemoryMappedTensor.ones_like(tensor, filename=file.name)
    """

    _filename: str | Path
    _handler: _FileHandler
    _clear: bool
    index: Any
    parent_shape: torch.Size

    def __new__(
        cls,
        tensor_or_file,
        *,
        dtype=None,
        shape=None,
        index=None,
        device=None,
        handler=None,
    ):
        if device is not None and torch.device(device).type != "cpu":
            raise ValueError(f"{cls} device must be cpu!")
        if isinstance(tensor_or_file, str):
            return cls.from_filename(
                tensor_or_file,
                dtype,
                shape,
                index,
            )
        elif handler is not None:
            return cls.from_handler(
                handler,
                dtype,
                shape,
                index,
            )
        return super().__new__(cls, tensor_or_file)

    def __init__(
        self, tensor_or_file, handler=None, dtype=None, shape=None, device=None
    ):
        ...

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def from_tensor(
        cls,
        input,
        *,
        filename=None,
        existsok=False,
        copy_existing=False,
        copy_data=True,
    ):
        """Creates a MemoryMappedTensor with the same content as another tensor.

        If the tensor is already a MemoryMappedTensor the original tensor is
        returned if the `filename` argument is `None` or if the two paths match.
        In all other cases, a new :class:`MemoryMappedTensor` is produced.

        Args:
            input (torch.Tensor): the tensor which content must be copied onto
                the MemoryMappedTensor.
            filename (path to a file): the path to the file where the tensor
                should be stored. If none is provided, a file handler is used
                instead.
            existsok (bool, optional): if ``True``, the file will overwrite
                an existing file. Defaults to ``False``.
            copy_existing (bool, optional): if ``True`` and the provided input
                is a MemoryMappedTensor with an associated filename, copying
                the content to the new location is permitted. Otherwise an
                exception is thown. This behaviour exists to prevent
                unadvertedly duplicating data on disk.
            copy_data (bool, optional): if ``True``, the content of the tensor
                will be copied on the storage. Defaults to ``True``.

        """
        if isinstance(input, MemoryMappedTensor):
            if (filename is None and input._filename is None) or (
                input._filename is not None
                and filename is not None
                and Path(filename).absolute() == Path(input.filename).absolute()
            ):
                # either location was not specified, or memmap is already in the
                # correct location, so just return the MemmapTensor unmodified
                return input
            elif not copy_existing and (
                input._filename is not None
                and filename is not None
                and Path(filename).absolute() != Path(input.filename).absolute()
            ):
                raise RuntimeError(
                    f"A filename was provided but the tensor already has a file associated "
                    f"({input.filename}). "
                    f"To copy the tensor onto the new location, pass copy_existing=True."
                )
        elif isinstance(input, np.ndarray):
            raise TypeError(
                "Convert input to torch.Tensor before calling MemoryMappedTensor.from_tensor."
            )
        if input.requires_grad:
            raise RuntimeError(
                "MemoryMappedTensor.from_tensor is incompatible with tensor.requires_grad."
            )
        shape = input.shape
        if filename is None:
            if input.dtype.is_floating_point:
                size = torch.finfo(input.dtype).bits // 8 * shape.numel()
            elif input.dtype.is_complex:
                raise ValueError(
                    "Complex-valued tensors are not supported by MemoryMappedTensor."
                )
            elif input.dtype == torch.bool:
                size = shape.numel()
            else:
                # assume integer
                size = torch.iinfo(input.dtype).bits // 8 * shape.numel()
            handler = _FileHandler(size)
            out = torch.frombuffer(memoryview(handler.buffer), dtype=input.dtype)
            out = out.view(shape)
            out = cls(out)
        else:
            handler = None
            if not existsok and os.path.exists(str(filename)):
                raise RuntimeError(f"The file {filename} already exists.")
            out = cls(
                torch.from_file(
                    str(filename), shared=True, dtype=input.dtype, size=shape.numel()
                ).view(input.shape)
            )
        out._handler = handler
        out._filename = filename
        out.index = None
        out.parent_shape = input.shape
        if copy_data:
            if isinstance(input, DTensor):
                input = input.full_tensor()
            out.copy_(input)
        return out

    @property
    def filename(self):
        """The filename of the tensor, if it has one.

        Raises an exception otherwise.
        """
        filename = self._filename
        if filename is None:
            raise RuntimeError("The MemoryMappedTensor has no file associated.")
        return filename

    @classmethod
    def empty_like(cls, input, *, filename=None):
        # noqa: D417
        """Creates a tensor with no content but the same shape and dtype as the input tensor.

        Args:
            input (torch.Tensor): the tensor to use as an example.

        Keyword Args:
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        return cls.from_tensor(
            torch.zeros((), dtype=input.dtype, device=input.device).expand_as(input),
            filename=filename,
            copy_data=False,
        )

    @classmethod
    def full_like(cls, input, fill_value, *, filename=None):
        # noqa: D417
        """Creates a tensor with a single content indicated by the `fill_value` argument, but the same shape and dtype as the input tensor.

        Args:
            input (torch.Tensor): the tensor to use as an example.
            fill_value (float or equivalent): content of the tensor.

        Keyword Args:
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        return cls.from_tensor(
            torch.zeros((), dtype=input.dtype, device=input.device).expand_as(input),
            filename=filename,
            copy_data=False,
        ).fill_(fill_value)

    @classmethod
    def zeros_like(cls, input, *, filename=None):
        # noqa: D417
        """Creates a tensor with a 0-filled content, but the same shape and dtype as the input tensor.

        Args:
            input (torch.Tensor): the tensor to use as an example.

        Keyword Args:
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        return cls.from_tensor(
            torch.zeros((), dtype=input.dtype, device=input.device).expand_as(input),
            filename=filename,
            copy_data=False,
        ).fill_(0.0)

    @classmethod
    def ones_like(cls, input, *, filename=None):
        # noqa: D417
        """Creates a tensor with a 1-filled content, but the same shape and dtype as the input tensor.

        Args:
            input (torch.Tensor): the tensor to use as an example.

        Keyword Args:
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        return cls.from_tensor(
            torch.ones((), dtype=input.dtype, device=input.device).expand_as(input),
            filename=filename,
            copy_data=False,
        ).fill_(1.0)

    @classmethod
    @overload
    def ones(cls, *size, dtype=None, device=None, filename=None):
        ...

    @classmethod
    @overload
    def ones(cls, shape, *, dtype=None, device=None, filename=None):
        ...

    @classmethod
    def ones(cls, *args, **kwargs):
        # noqa: D417
        """Creates a tensor with a 1-filled content, specific shape, dtype and filename.

        Args:
            shape (integers or torch.Size): the shape of the tensor.

        Keyword Args:
            dtype (torch.dtype): the dtype of the tensor.
            device (torch.device): the device of the tensor. Only `None` and `"cpu"`
                are accepted, any other device will raise an exception.
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        shape, device, dtype, _, filename = _proc_args_const(*args, **kwargs)
        if device is not None:
            device = torch.device(device)
            if device.type != "cpu":
                raise RuntimeError("Only CPU tensors are supported.")
        result = torch.ones((), dtype=dtype, device=device)
        if shape:
            if isinstance(shape[0], (list, tuple)) and len(shape) == 1:
                shape = torch.Size(shape[0])
            else:
                shape = torch.Size(shape)
            result = result.expand(shape)
        return cls.from_tensor(
            result,
            filename=filename,
        )

    @classmethod
    @overload
    def zeros(cls, *size, dtype=None, device=None, filename=None):
        ...

    @classmethod
    @overload
    def zeros(cls, shape, *, dtype=None, device=None, filename=None):
        ...

    @classmethod
    def zeros(cls, *args, **kwargs):
        # noqa: D417
        """Creates a tensor with a 0-filled content, specific shape, dtype and filename.

        Args:
            shape (integers or torch.Size): the shape of the tensor.

        Keyword Args:
            dtype (torch.dtype): the dtype of the tensor.
            device (torch.device): the device of the tensor. Only `None` and `"cpu"`
                are accepted, any other device will raise an exception.
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        shape, device, dtype, _, filename = _proc_args_const(*args, **kwargs)
        if device is not None:
            device = torch.device(device)
            if device.type != "cpu":
                raise RuntimeError("Only CPU tensors are supported.")
        result = torch.zeros((), dtype=dtype, device=device)
        if shape:
            if isinstance(shape[0], (list, tuple)) and len(shape) == 1:
                shape = torch.Size(shape[0])
            else:
                shape = torch.Size(shape)
            result = result.expand(shape)
        result = cls.from_tensor(
            result,
            filename=filename,
        )
        return result

    @classmethod
    @overload
    def empty(cls, *size, dtype=None, device=None, filename=None):
        ...

    @classmethod
    @overload
    def empty(cls, shape, *, dtype=None, device=None, filename=None):
        ...

    @classmethod
    def empty(cls, *args, **kwargs):
        # noqa: D417
        """Creates a tensor with empty content, specific shape, dtype and filename.

        Args:
            shape (integers or torch.Size): the shape of the tensor.

        Keyword Args:
            dtype (torch.dtype): the dtype of the tensor.
            device (torch.device): the device of the tensor. Only `None` and `"cpu"`
                are accepted, any other device will raise an exception.
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        shape, device, dtype, _, filename = _proc_args_const(*args, **kwargs)
        if device is not None:
            device = torch.device(device)
            if device.type != "cpu":
                raise RuntimeError("Only CPU tensors are supported.")
        result = torch.zeros((), dtype=dtype, device=device)
        if shape:
            if isinstance(shape[0], (list, tuple)) and len(shape) == 1:
                shape = torch.Size(shape[0])
            else:
                shape = torch.Size(shape)
            result = result.expand(shape)
        result = cls.from_tensor(result, filename=filename)
        return result

    @classmethod
    @overload
    def full(cls, *size, fill_value, dtype=None, device=None, filename=None):
        ...

    @classmethod
    @overload
    def full(cls, shape, *, fill_value, dtype=None, device=None, filename=None):
        ...

    @classmethod
    def full(cls, *args, **kwargs):
        # noqa: D417
        """Creates a tensor with a single content specified by `fill_value`, specific shape, dtype and filename.

        Args:
            shape (integers or torch.Size): the shape of the tensor.

        Keyword Args:
            fill_value (float or equivalent): content of the tensor.
            dtype (torch.dtype): the dtype of the tensor.
            device (torch.device): the device of the tensor. Only `None` and `"cpu"`
                are accepted, any other device will raise an exception.
            filename (path or equivalent): the path to the file, if any. If none
                is provided, a handler is used.
        """
        shape, device, dtype, fill_value, filename = _proc_args_const(*args, **kwargs)
        if device is not None:
            device = torch.device(device)
            if device.type != "cpu":
                raise RuntimeError("Only CPU tensors are supported.")
        result = torch.zeros((), dtype=dtype, device=device).fill_(fill_value)
        if shape:
            if isinstance(shape[0], (list, tuple)) and len(shape) == 1:
                shape = torch.Size(shape[0])
            else:
                shape = torch.Size(shape)
            result = result.expand(shape)
        return cls.from_tensor(result, filename=filename)

    @classmethod
    def from_filename(cls, filename, dtype, shape, index=None):
        # noqa: D417
        """Loads a MemoryMappedTensor from a given filename.

        Args:
            filename (path or equivalent): the path to the file.
            dtype (torch.dtype): the dtype of the tensor.
            shape (integers or torch.Size): the shape of the tensor.
            index (torch-compatible index type): an index to use to build the
                tensor.

        """
        shape = torch.Size(shape)
        tensor = torch.from_file(
            str(filename), shared=True, dtype=dtype, size=shape.numel()
        ).view(shape)
        if index is not None:
            tensor = tensor[index]
        out = cls(tensor)
        out._filename = filename
        out._handler = None
        out.index = index
        out.parent_shape = shape
        return out

    @classmethod
    def from_handler(cls, handler, dtype, shape, index):
        # noqa: D417
        """Loads a MemoryMappedTensor from a given handler.

        Args:
            handler (compatible file handler): the handler for the tensor.
            dtype (torch.dtype): the dtype of the tensor.
            shape (integers or torch.Size): the shape of the tensor.
            index (torch-compatible index type): an index to use to build the
                tensor.

        """
        shape = torch.Size(shape)
        out = torch.frombuffer(memoryview(handler.buffer), dtype=dtype)
        out = torch.reshape(out, shape)
        if index is not None:
            out = out[index]
        out = cls(out)
        out._filename = None
        out._handler = handler
        out.index = index
        out.parent_shape = shape
        return out

    @property
    def _tensor(self):
        # for bc-compatibility with MemmapTensor, to be deprecated in v0.4
        return self

    def __setstate__(self, state):
        if "filename" in state:
            self.__dict__ = type(self).from_filename(**state).__dict__
        else:
            self.__dict__ = type(self).from_handler(**state).__dict__

    def __getstate__(self):
        if getattr(self, "_handler", None) is not None:
            return {
                "handler": self._handler,
                "dtype": self.dtype,
                "shape": self.parent_shape,
                "index": self.index,
            }
        elif getattr(self, "_filename", None) is not None:
            return {
                "filename": self._filename,
                "dtype": self.dtype,
                "shape": self.parent_shape,
                "index": self.index,
            }
        else:
            raise RuntimeError("Could not find handler or filename.")

    def __reduce_ex__(self, protocol):
        return self.__reduce__()

    def __reduce__(self):
        if getattr(self, "_handler", None) is not None:
            return type(self).from_handler, (
                self._handler,
                self.dtype,
                self.parent_shape,
                self.index,
            )
        elif getattr(self, "_filename", None) is not None:
            return type(self).from_filename, (
                self._filename,
                self.dtype,
                self.parent_shape,
                self.index,
            )
        else:
            raise RuntimeError("Could not find handler or filename.")

    @implement_for("torch", "2.0", None)
    def __getitem__(self, item):
        try:
            out = super().__getitem__(item)
        except ValueError as err:
            if "is unbound" in str(err):
                raise ValueError(
                    "Using first class dimension indices with MemoryMappedTensor "
                    "isn't supported at the moment."
                ) from err
            raise
        if out.untyped_storage().data_ptr() == self.untyped_storage().data_ptr():
            out = MemoryMappedTensor(out)
            out._handler = self._handler
            out._filename = self._filename
            out.index = item
            out.parent_shape = self.parent_shape
        return out

    @implement_for("torch", None, "2.0")
    def __getitem__(self, item):  # noqa: F811
        try:
            out = super().__getitem__(item)
        except ValueError as err:
            if "is unbound" in str(err):
                raise ValueError(
                    "Using first class dimension indices with MemoryMappedTensor "
                    "isn't supported at the moment."
                ) from err
            raise
        if out.storage().data_ptr() == self.storage().data_ptr():
            out = MemoryMappedTensor(out)
            out._handler = self._handler
            out._filename = self._filename
            out.index = item
            out.parent_shape = self.parent_shape
        return out


#####################
# File handler
# borrowed from mp.heap

if sys.platform == "win32":
    import _winapi

    class _FileHandler:
        _rand = tempfile._RandomNameSequence()

        def __init__(self, size):
            self.size = size
            for _ in range(100):
                name = "pym-%d-%s" % (os.getpid(), next(self._rand))
                buf = mmap.mmap(-1, size, tagname=name)
                if _winapi.GetLastError() == 0:
                    break
                # We have reopened a preexisting mmap.
                buf.close()
            else:
                raise FileExistsError("Cannot find name for new mmap")
            self.name = name
            self.buffer = buf
            self._state = (self.size, self.name)

        def __getstate__(self):
            from multiprocessing.context import assert_spawning

            assert_spawning(self)
            return self._state

        def __setstate__(self, state):
            self.size, self.name = self._state = state
            # Reopen existing mmap
            self.buffer = mmap.mmap(-1, self.size, tagname=self.name)
            # XXX Temporarily preventing buildbot failures while determining
            # XXX the correct long-term fix. See issue 23060
            # assert _winapi.GetLastError() == _winapi.ERROR_ALREADY_EXISTS

else:

    class _FileHandler:
        if sys.platform == "linux":
            _dir_candidates = ["/dev/shm"]
        else:
            _dir_candidates = []

        def __init__(self, size, fd=-1):
            self.size = size
            self.fd = fd
            if fd == -1:
                self.fd, name = tempfile.mkstemp(
                    prefix="pym-%d-" % os.getpid(), dir=self._choose_dir(size)
                )
                os.unlink(name)
                util.Finalize(self, os.close, (self.fd,))
                os.ftruncate(self.fd, size)
            self.buffer = mmap.mmap(self.fd, self.size)

        def _choose_dir(self, size):
            # Choose a non-storage backed directory if possible,
            # to improve performance
            for d in self._dir_candidates:
                st = os.statvfs(d)
                if st.f_bavail * st.f_frsize >= size:  # enough free space?
                    return d
            return util.get_temp_dir()

    def _reduce_handler(handler):
        if handler.fd == -1:
            raise ValueError(
                "Handler is unpicklable because "
                "forking was enabled when it was created"
            )
        return _rebuild_handler, (handler.size, reduction.DupFd(handler.fd))

    def _rebuild_handler(size, dupfd):
        detached = dupfd.detach()
        return _FileHandler(size, detached)

    reduction.register(_FileHandler, _reduce_handler)


def _reduce_memmap(memmap_tensor):
    return memmap_tensor.__reduce__()


ForkingPickler.register(MemoryMappedTensor, _reduce_memmap)


def _proc_args_const(*args, **kwargs):
    if len(args) > 0:
        # then the first (or the N first) args are the shape
        if len(args) == 1 and not isinstance(args[0], int):
            shape = torch.Size(args[0])
        else:
            shape = torch.Size(args)
    else:
        # we should have a "shape" keyword arg
        shape = kwargs.pop("shape", None)
        if shape is None:
            raise TypeError("Could not find the shape argument in the arguments.")
        shape = torch.Size(shape)
    return (
        shape,
        kwargs.pop("device", None),
        kwargs.pop("dtype", None),
        kwargs.pop("fill_value", None),
        kwargs.pop("filename", None),
    )
