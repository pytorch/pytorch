import torch
from .._utils import _dummy_type
from typing import Union

if not hasattr(torch._C, "_CudaGdsFileBase"):
    # Define dummy base classes
    torch._C.__dict__["_CudaGdsFileBase"] = _dummy_type("_CudaGdsFileBase")

if not hasattr(torch._C, "_gds_register_buffer"):
    # Define dummy base classes
    torch._C.__dict__["_gds_register_buffer"] = _dummy_type("_gds_register_buffer")

if not hasattr(torch._C, "_gds_deregister_buffer"):
    # Define dummy base classes
    torch._C.__dict__["_gds_deregister_buffer"] = _dummy_type("_gds_deregister_buffer")


def gds_register_buffer(t: Union[torch.Storage, torch.Tensor]):
    """Registers a buffer.

    Args:
        t (Tensor or Storage): Buffer to register.
    """
    torch._C._gds_register_buffer(t)


def gds_deregister_buffer(t: Union[torch.Storage, torch.Tensor]):
    """Registers a buffer.

    Args:
        t (Tensor or Storage): Buffer to register.
    """
    torch._C._gds_deregister_buffer(t)


class GdsFile(torch._C._CudaGdsFileBase):
    r"""Wrapper around cuFile.

    cuFile is a file-like interface to the GPUDirect Storage (GDS) API.

    Args:
        filename (str): Name of the file to open.
        mode (str): Mode to open the file in.
    
    .. _CUDA GPUDirect Storage Documentation:
    <TODO: fill in link>
    """

    def __new__(cls, filename: str, mode: str):
        return super().__new__(cls, filename, mode)
    
    def load_storage(self, storage: torch.Storage, offset: int = 0):
        """Loads data from the file into the storage.

        ``storage.nbytes()`` of data will be loaded from the file at ``offset``
        into the storage.

        Args:
            storage (torch.Storage): Storage to load data into.
            offset (int, optional): Offset into the file to start loading from.
        """
        self.load_storage(storage, offset)
    
    def save_storage(self, storage: torch.Storage, offset: int = 0):
        """Saves data from the storage into the file.

        All bytes of the storage will be written to the file at ``offset``.

        Args:
            storage (torch.Storage): Storage to save data from.
            offset (int, optional): Offset into the file to start saving to.
        """
        self.save_storage(storage, offset)
    
    def load_tensor(self, tensor: torch.Tensor, offset: int = 0):
        """Loads data from the file into the tensor.

        ``tensor.nbytes()`` of data will be loaded from the file at ``offset``
        into the tensor.

        Args:
            tensor (torch.Tensor): Tensor to load data into.
            offset (int, optional): Offset into the file to start loading from.
        """
        self.load_tensor(tensor, offset)

    def save_tensor(self, tensor: torch.Tensor, offset: int = 0):
        """Saves data from the tensor into the file.

        All bytes of the tensor will be written to the file at ``offset``.

        Args:
            tensor (torch.Tensor): Tensor to save data from.
            offset (int, optional): Offset into the file to start saving to.
        """
        self.save_tensor(tensor, offset)
