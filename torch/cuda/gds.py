import os
from typing import Callable

import torch


def _dummy_fn(name: str) -> Callable:
    def fn(*args, **kwargs):
        raise RuntimeError(f"{name} is not supported on this platform")

    return fn


if not hasattr(torch._C, "gds_register_buffer"):
    assert not hasattr(torch._C, "gds_deregister_buffer")
    assert not hasattr(torch._C, "gds_register_handle")
    assert not hasattr(torch._C, "gds_deregister_handle")
    assert not hasattr(torch._C, "gds_load_storage")
    assert not hasattr(torch._C, "gds_save_storage")
    # Define functions
    torch._C.__dict__["gds_register_buffer"] = _dummy_fn("gds_register_buffer")
    torch._C.__dict__["gds_deregister_buffer"] = _dummy_fn("gds_deregister_buffer")
    torch._C.__dict__["gds_register_handle"] = _dummy_fn("gds_register_handle")
    torch._C.__dict__["gds_deregister_handle"] = _dummy_fn("gds_deregister_handle")
    torch._C.__dict__["gds_load_storage"] = _dummy_fn("gds_register_handle")
    torch._C.__dict__["gds_save_storage"] = _dummy_fn("gds_register_handle")

if not hasattr(torch._C, "_gds_deregister_buffer"):
    # Define dummy base classes
    torch._C.__dict__["_gds_deregister_buffer"] = _dummy_fn("_gds_deregister_buffer")


def gds_register_buffer(t: torch.UntypedStorage) -> None:
    """Registers a buffer.

    Args:
        t (Tensor or Storage): Buffer to register.
    """
    torch._C.gds_register_buffer(t)


def gds_deregister_buffer(t: torch.UntypedStorage) -> None:
    """Registers a buffer.

    Args:
        t (Tensor or Storage): Buffer to register.
    """
    torch._C.gds_deregister_buffer(t)


class GdsFile:
    r"""Wrapper around cuFile.

    cuFile is a file-like interface to the GPUDirect Storage (GDS) API.

    Args:
        filename (str): Name of the file to open.
        flags (int): Flags to pass to os.open when opening the file. ``os.O_DIRECT`` will
            be added automatically.

    .. _CUDA GPUDirect Storage Documentation:
        https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufile-io-api
    """

    def __init__(self, filename: str, flags: int):
        self.filename = filename
        self.flags = flags
        self.fd = os.open(filename, flags | os.O_DIRECT)
        self.register_handle()

    def register_handle(self) -> None:
        """Registers file descriptor to cuFile Driver.

        Args:
            handle (int): Handle to register.
        """
        self.handle = torch._C.gds_register_handle(self.fd)

    def deregister_handle(self) -> None:
        """Deregisters file descriptor from cuFile Driver.

        Args:
            handle (int): Handle to register.
        """
        assert (
            self.handle is not None
        ), "Cannot deregister a handle that is not registered."
        torch._C.gds_deregister_handle(self.handle)
        self.handle = None

    def load_storage(self, storage: torch.UntypedStorage, offset: int = 0) -> None:
        """Loads data from the file into the storage.

        ``storage.nbytes()`` of data will be loaded from the file at ``offset``
        into the storage.

        Args:
            storage (torch.UntypedStorage): Storage to load data into.
            offset (int, optional): Offset into the file to start loading from.
        """
        torch._C.gds_load_storage(self.handle, storage, offset)

    def save_storage(self, storage: torch.UntypedStorage, offset: int = 0) -> None:
        """Saves data from the storage into the file.

        All bytes of the storage will be written to the file at ``offset``.

        Args:
            storage (torch.UntypedStorage): Storage to save data from.
            offset (int, optional): Offset into the file to start saving to.
        """
        torch._C.gds_save_storage(self.handle, storage, offset)
