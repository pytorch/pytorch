# mypy: allow-untyped-defs
"""
Python stubs for backend-specific distributed components.

Since _C._distributed_c10d always exists now, this module only provides
stubs for backend-specific functionality that may not be available in all builds
(e.g., NCCL, UCC, MPI, Gloo, etc.).
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from datetime import timedelta

import torch


# Store classes
class HashStore:
    """Stub HashStore for builds without this functionality."""

    def __init__(self, *args, **kwargs):
        self._data = {}

    def set(self, key: str, value: str):
        self._data[key] = value

    def get(self, key: str) -> bytes:
        return self._data.get(key, "").encode()


# Backend-specific process group stubs
class ProcessGroupMPI:
    """Stub ProcessGroupMPI for non-MPI builds."""

    def __init__(self, *args, **kwargs):
        pass


class ProcessGroupNCCL:
    """Stub ProcessGroupNCCL for non-NCCL builds."""

    def __init__(self, *args, **kwargs):
        pass


class ProcessGroupGloo:
    """Stub ProcessGroupGloo for non-Gloo builds."""

    def __init__(self, *args, **kwargs):
        pass


class ProcessGroupUCC:
    """Stub ProcessGroupUCC for non-UCC builds."""

    def __init__(self, *args, **kwargs):
        pass


class ProcessGroupXCCL:
    """Stub ProcessGroupXCCL for non-XCCL builds."""

    def __init__(self, *args, **kwargs):
        pass


class _ProcessGroupWrapper:
    """Stub _ProcessGroupWrapper for non-Gloo builds."""

    def __init__(self, process_group, *args, **kwargs):
        self._process_group = process_group

    def __getattr__(self, name):
        return getattr(self._process_group, name)


# NCCL-specific function stubs
_DEFAULT_PG_NCCL_TIMEOUT: Optional[timedelta] = None


def _hash_tensors(tensors):
    """Stub function to hash tensors - returns dummy hash."""
    return 0


def _dump_nccl_trace_json(
    includeCollectives: Optional[bool] = None, onlyActive: Optional[bool] = None
) -> bytes:
    """Stub function that returns empty JSON trace."""
    return b"{}"


def _dump_nccl_trace(
    includeCollectives: Optional[bool] = None,
    includeStackTraces: Optional[bool] = None,
    onlyActive: Optional[bool] = None,
) -> bytes:
    """Stub function that returns empty pickle trace."""
    return b""


# NVSHMEM/SymmetricMemory stubs
def _is_nvshmem_available() -> bool:
    """Stub function that returns False indicating NVSHMEM is not available."""
    return False


def _nvshmemx_cumodule_init(module: int) -> None:
    """Stub function for NVSHMEM CU module initialization."""


class _SymmetricMemory:
    """Stub _SymmetricMemory class for builds without this functionality."""

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def empty_strided_p2p(cls, size, stride, dtype, device, group_name=None):
        """Stub that returns a regular tensor."""
        return torch.empty(size, dtype=dtype, device=device)

    @classmethod
    def rendezvous(cls, tensor, group_name=None):
        """Stub that returns None."""
        return None

    @classmethod
    def set_group_info(cls, *args, **kwargs):
        """Stub that does nothing."""

    @classmethod
    def set_backend(cls, name):
        """Stub that does nothing."""

    @classmethod
    def get_backend(cls, device):
        """Stub that returns None."""
        return None

    @classmethod
    def has_multicast_support(cls, device_type, device_index):
        """Stub that returns False."""
        return False
