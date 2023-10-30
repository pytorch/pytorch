# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import mmap
import os

import sys
import tempfile
from multiprocessing import util
from multiprocessing.context import reduction
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch.multiprocessing.reductions import ForkingPickler


class MemoryMappedTensor(torch.Tensor):
    _filename: str | Path
    _handler: FileHandler
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
    def from_tensor(cls, tensor, *, dir=None, filename=None):
        if isinstance(tensor, MemoryMappedTensor):
            if dir is None and (
                filename is None
                or Path(filename).absolute() == Path(tensor._filename).absolute()
            ):
                # either location was not specified, or memmap is already in the
                # correct location, so just return the MemmapTensor unmodified
                return tensor
        elif isinstance(tensor, np.ndarray):
            raise TypeError(
                "Convert input to torch.Tensor before calling MemoryMappedTensor.from_tensor."
            )
        if tensor.requires_grad:
            raise RuntimeError(
                "MemoryMappedTensor.from_tensor is incompatible with tensor.requires_grad."
            )
        shape = tensor.shape
        if filename is None:
            if tensor.dtype.is_floating_point:
                size = torch.finfo(tensor.dtype).bits // 8 * shape.numel()
            elif tensor.dtype.is_complex:
                raise ValueError(
                    "Complex-valued tensors are not supported by MemoryMappedTensor."
                )
            elif tensor.dtype == torch.bool:
                size = shape.numel()
            else:
                # assume integer
                size = torch.iinfo(tensor.dtype).bits // 8 * shape.numel()
            handler = FileHandler(size)
            out = torch.frombuffer(memoryview(handler.buffer), dtype=tensor.dtype)
            out = torch.reshape(out, shape)
            out = cls(out)
        else:
            handler = None
            out = cls(
                torch.from_file(
                    str(filename), shared=True, dtype=tensor.dtype, size=shape.numel()
                ).view(tensor.shape)
            )
        out._handler = handler
        out._filename = filename
        out.index = None
        out.parent_shape = tensor.shape
        out.copy_(tensor)
        return out

    @property
    def filename(self):
        filename = self._filename
        if filename is None:
            raise RuntimeError("The MemoryMappedTensor has no file associated.")
        return filename

    @classmethod
    def empty_like(cls, tensor, *, filename=None):
        return cls.from_tensor(
            torch.zeros((), dtype=tensor.dtype, device=tensor.device).expand_as(tensor),
            filename=filename,
        )

    @classmethod
    def from_filename(cls, filename, dtype, shape, index=None):
        tensor = torch.from_file(
            filename, shared=True, dtype=dtype, size=shape.numel()
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
        if out.data_ptr() == self.data_ptr():
            out = MemoryMappedTensor(out)
            out._handler = self._handler
            out._filename = self._filename
            out.index = item
            out.parent_shape = self.parent_shape
        return out


class FileHandler:
    if sys.platform == "linux":
        _dir_candidates = ["/dev/shm"]
    else:
        _dir_candidates = []

    def __init__(self, size, fd=-1, filename=None):
        # borrowed from mp.heap
        self.size = size
        # if filename is None:
        if fd == -1:
            self.fd, name = tempfile.mkstemp(
                prefix="pym-%d-" % os.getpid(), dir=self._choose_dir(size)
            )
            # self.filename = name
            os.unlink(name)
            util.Finalize(self, os.close, (self.fd,))
            os.ftruncate(self.fd, size)
        else:
            self.fd = fd
        # else:
        #     self.filename = filename
        self.buffer = mmap.mmap(self.fd, self.size)

    def _choose_dir(self, size):
        # Choose a non-storage backed directory if possible,
        # to improve performance
        for d in self._dir_candidates:
            st = os.statvfs(d)
            if st.f_bavail * st.f_frsize >= size:  # enough free space?
                return d
        tmpdir = util.get_temp_dir()
        return tmpdir


def reduce_handler(handler):
    if handler.fd == -1:
        raise ValueError(
            "Handler is unpicklable because " "forking was enabled when it was created"
        )
    return rebuild_handler, (handler.size, reduction.DupFd(handler.fd))


def rebuild_handler(size, dupfd):
    detached = dupfd.detach()
    return FileHandler(size, detached)


reduction.register(FileHandler, reduce_handler)


def reduce_memmap(memmap_tensor):
    return memmap_tensor.__reduce__()


ForkingPickler.register(MemoryMappedTensor, reduce_memmap)
