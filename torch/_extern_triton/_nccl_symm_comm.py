# Owner(s): ["oncall: distributed"]
"""
Python wrapper for NCCL symmetric memory communication.

This module provides a high-level Python interface for the NCCLSymmComm class
that handles NCCL initialization, symmetric memory allocation, window registration,
and device communicator creation.

Usage:
    from torch._extern_triton._nccl_symm_comm import NCCLSymmComm

    # Initialize communicator (call after process group is initialized)
    comm = NCCLSymmComm(group_name="0", buffer_size=1024*1024, device_idx=0)

    # Get pointers for Triton kernels
    ctx_ptr = comm.get_context_ptr()  # SymmContext pointer
    buffer_ptr = comm.get_buffer_ptr()  # Local buffer pointer
"""

import logging

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


def is_nccl_symm_mem_available() -> bool:
    """Check if NCCL symmetric memory is available."""
    try:
        import torch.distributed._symmetric_memory as symm_mem

        # Check if set_backend with NCCL is available
        return hasattr(symm_mem, "set_backend")
    except (AttributeError, RuntimeError, ImportError):
        return False


class NCCLSymmComm:
    """
    High-level Python wrapper for NCCL symmetric memory communication.

    This class wraps the C++ NCCLSymmComm class and provides a Pythonic interface
    for initializing NCCL symmetric memory communication and obtaining pointers
    that can be passed to Triton kernels.

    Attributes:
        group_name: Name of the process group
        buffer_size: Size of the symmetric buffer in bytes
        device_idx: CUDA device index
    """

    def __init__(
        self,
        group_name: str | None = None,
        buffer_size: int = 1024 * 1024,
        device_idx: int | None = None,
    ):
        """
        Initialize NCCL symmetric memory communication.

        Args:
            group_name: Name of the process group. If None, uses the default group.
            buffer_size: Size of the symmetric buffer in bytes. Default is 1MB.
            device_idx: CUDA device index. If None, uses current device.

        Raises:
            RuntimeError: If NCCL symmetric memory is not available or
                          if distributed is not initialized.
        """
        if not is_nccl_symm_mem_available():
            raise RuntimeError(
                "NCCL symmetric memory is not available. "
                "Requires NCCL >= 2.28.9 with symmetric memory support."
            )

        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before creating NCCLSymmComm. "
                "Call dist.init_process_group() first."
            )

        # Resolve group name
        if group_name is None:
            world_group = dist.group.WORLD
            if world_group is not None:
                group_name = world_group.group_name
            else:
                group_name = "0"  # Default group name

        # Resolve device index
        if device_idx is None:
            device_idx = torch.cuda.current_device()

        self._group_name = group_name
        self._buffer_size = buffer_size
        self._device_idx = device_idx
        self._comm = None

        # Create the C++ object
        try:
            from torch._C._distributed_c10d import (  # type: ignore[attr-defined]
                NCCLSymmComm as _NCCLSymmComm,
            )

            self._comm = _NCCLSymmComm(group_name, buffer_size, device_idx)
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import NCCLSymmComm: {e}. "
                "Make sure PyTorch is built with NCCL symmetric memory support."
            ) from e

    @property
    def group_name(self) -> str:
        """Get the process group name."""
        return self._group_name

    @property
    def buffer_size(self) -> int:
        """Get the buffer size in bytes."""
        if self._comm is None:
            return self._buffer_size
        return self._comm.get_buffer_size()

    @property
    def device_idx(self) -> int:
        """Get the device index."""
        if self._comm is None:
            return self._device_idx
        return self._comm.get_device_idx()

    @property
    def rank(self) -> int:
        """Get the rank of this process."""
        if self._comm is None:
            return dist.get_rank()
        return self._comm.get_rank()

    @property
    def world_size(self) -> int:
        """Get the world size (number of processes)."""
        if self._comm is None:
            return dist.get_world_size()
        return self._comm.get_world_size()

    def get_context_ptr(self) -> int:
        """
        Get pointer to the device-side SymmContext.

        This pointer can be passed to Triton kernels as an int64 value.

        Returns:
            int64: Device pointer to NCCLSymmContext
        """
        if self._comm is None:
            raise RuntimeError("NCCLSymmComm not initialized")
        return self._comm.get_context_ptr()

    def get_buffer_ptr(self) -> int:
        """
        Get pointer to the local symmetric buffer.

        This pointer can be passed to Triton kernels as an int64 value.

        Returns:
            int64: Device pointer to the symmetric buffer
        """
        if self._comm is None:
            raise RuntimeError("NCCLSymmComm not initialized")
        return self._comm.get_buffer_ptr()

    def get_buffer_as_tensor(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Get the local buffer as a PyTorch tensor.

        This creates a tensor view of the symmetric memory buffer.
        Note: The tensor shares memory with the symmetric buffer.

        Args:
            dtype: Data type for the tensor. Default is float32.

        Returns:
            torch.Tensor: Tensor view of the symmetric buffer
        """
        if self._comm is None:
            raise RuntimeError("NCCLSymmComm not initialized")

        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = self.buffer_size // element_size

        # Create tensor from existing memory using storage
        # Note: This is a lower-level approach that works with device memory
        storage = torch.cuda.FloatStorage._new_shared_cuda(
            num_elements,
            self.get_buffer_ptr(),
            torch.device("cuda", self.device_idx),
        )
        tensor = torch.tensor(
            storage, dtype=dtype, device=torch.device("cuda", self.device_idx)
        )
        return tensor.view(num_elements)

    def __repr__(self) -> str:
        return (
            f"NCCLSymmComm(group_name='{self.group_name}', "
            f"buffer_size={self.buffer_size}, "
            f"device_idx={self.device_idx}, "
            f"rank={self.rank}, "
            f"world_size={self.world_size})"
        )
