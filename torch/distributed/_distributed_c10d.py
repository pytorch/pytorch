# mypy: allow-untyped-defs
"""
Centralized module for importing and re-exporting torch._C._distributed_c10d components.
This module provides fallback stubs when distributed components are not available.

IMPORTANT PATTERN:
Never access torch._C._distributed_c10d directly in code. Always import from and use
torch.distributed._distributed_c10d which is guaranteed to have all functions available
either from the C extension (when distributed is built) or from Python stubs (when not built).

Example:
    # WRONG: torch._C._distributed_c10d._set_global_rank(rank)
    # RIGHT:
    from torch.distributed._distributed_c10d import _set_global_rank
    _set_global_rank(rank)

This ensures code works regardless of whether distributed components are available.

IMPORTANT: This file should only have ONE try-catch block that imports torch._C._distributed_c10d.
All other imports should be handled in the if HAS_DISTRIBUTED: ... else: ... block.
"""

import sys
from datetime import timedelta
from typing import Optional

import torch


# Single minimal try-catch block to import the C extension
try:
    import torch._C._distributed_c10d as _C
except (ImportError, AttributeError):
    _C = None  # type: ignore[assignment]

# Set HAS_DISTRIBUTED based on whether C extension is available
HAS_DISTRIBUTED = _C is not None

if HAS_DISTRIBUTED:
    from torch._C._distributed_c10d import *  # noqa: F403,F401
else:
    from torch.distributed._C_stubs import *  # noqa: F403,F401

# Handle optional components that may not be available in all builds
# Import NCCL-specific components if available
try:
    from torch._C._distributed_c10d import _DEFAULT_PG_NCCL_TIMEOUT
except ImportError:
    _DEFAULT_PG_NCCL_TIMEOUT: Optional[timedelta] = None  # type: ignore[no-redef]

# Import optional components that may not be available in all builds
try:
    from torch._C._distributed_c10d import FakeProcessGroup
except ImportError:
    FakeProcessGroup: Optional[type] = None  # type: ignore[misc,no-redef]

# Import platform-specific components
if sys.platform != "win32":
    try:
        from torch._C._distributed_c10d import HashStore
    except ImportError:
        # Use the stub version if not available
        pass
else:
    # Provide HashStore stub for Windows (override the stub version)
    class HashStore(Store):  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            super().__init__()


# Import NVSHMEM and symmetric memory components if available
try:
    from torch._C._distributed_c10d import (
        _is_nvshmem_available,
        _nvshmemx_cumodule_init,
        _SymmetricMemory,
    )
except ImportError:
    # Provide fallback stubs if not available (override stub versions)
    def _is_nvshmem_available() -> bool:  # type: ignore[no-redef]
        return False

    def _nvshmemx_cumodule_init(module: int) -> None:  # type: ignore[no-redef]
        """Mock function for NVSHMEM CU module initialization - does nothing in non-distributed builds."""

    class _SymmetricMemory:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def empty_strided_p2p(cls, size, stride, dtype, device, group_name=None):
            return torch.empty(size, dtype=dtype, device=device)

        @classmethod
        def rendezvous(cls, tensor, group_name=None):
            return None

        @classmethod
        def set_group_info(cls, *args, **kwargs):
            pass

        @classmethod
        def set_backend(cls, name):
            pass

        @classmethod
        def get_backend(cls, device):
            return None

        @classmethod
        def has_multicast_support(cls, device_type, device_index):
            return False


# Check availability of specific process group implementations
# These variables track whether specific backends are available in the C extension
_MPI_AVAILABLE = True
_NCCL_AVAILABLE = True
_GLOO_AVAILABLE = True
_UCC_AVAILABLE = True
_XCCL_AVAILABLE = True

if HAS_DISTRIBUTED:
    # Test if each process group type is actually available in the C extension
    try:
        from torch._C._distributed_c10d import ProcessGroupMPI  # noqa: F401
    except ImportError:
        _MPI_AVAILABLE = False

    try:
        from torch._C._distributed_c10d import ProcessGroupNCCL  # noqa: F401
    except ImportError:
        _NCCL_AVAILABLE = False

    try:
        from torch._C._distributed_c10d import (  # noqa: F401
            _ProcessGroupWrapper,
            ProcessGroupGloo,
        )
    except ImportError:
        _GLOO_AVAILABLE = False

    try:
        from torch._C._distributed_c10d import ProcessGroupUCC  # noqa: F401
    except ImportError:
        _UCC_AVAILABLE = False

    try:
        from torch._C._distributed_c10d import ProcessGroupXCCL  # noqa: F401
    except ImportError:
        _XCCL_AVAILABLE = False
else:
    # When distributed is not available, none of the specific backends are available
    _MPI_AVAILABLE = False
    _NCCL_AVAILABLE = False
    _GLOO_AVAILABLE = False
    _UCC_AVAILABLE = False
    _XCCL_AVAILABLE = False


# Provide backwards compatibility by making all symbols available at module level
__all__ = [
    # Basic components
    "_broadcast_coalesced",
    "_compute_bucket_assignment_by_size",
    "_ControlCollectives",
    "_DEFAULT_FIRST_BUCKET_BYTES",
    "_DEFAULT_PG_TIMEOUT",
    "_DEFAULT_PG_NCCL_TIMEOUT",
    "_make_nccl_premul_sum",
    "_register_builtin_comm_hook",
    "_register_comm_hook",
    "_StoreCollectives",
    "_test_python_store",
    "_verify_params_across_processes",
    "_allow_inflight_collective_as_graph_input",
    "_register_work",
    "_set_allow_inflight_collective_as_graph_input",
    "_is_nvshmem_available",
    "_nvshmemx_cumodule_init",
    "_SymmetricMemory",
    "Backend",
    "BuiltinCommHookType",
    "DebugLevel",
    "FakeProcessGroup",
    "FileStore",
    "get_debug_level",
    "GradBucket",
    "HashStore",
    "Logger",
    "PrefixStore",
    "ProcessGroup",
    "Reducer",
    "ReduceOp",
    "set_debug_level",
    "set_debug_level_from_env",
    "Store",
    "TCPStore",
    "Work",
    "FakeWork",
    # Additional distributed_c10d components
    "_DistributedBackendOptions",
    "_register_process_group",
    "_resolve_process_group",
    "_unregister_all_process_groups",
    "_unregister_process_group",
    "_current_process_group",
    "_set_process_group",
    "_WorkerServer",
    "AllgatherOptions",
    "AllreduceCoalescedOptions",
    "AllreduceOptions",
    "AllToAllOptions",
    "BarrierOptions",
    "BroadcastOptions",
    "GatherOptions",
    "ReduceOptions",
    "ReduceScatterOptions",
    "ScatterOptions",
    # Process group implementations
    "ProcessGroupMPI",
    "ProcessGroupNCCL",
    "ProcessGroupGloo",
    "ProcessGroupUCC",
    "ProcessGroupXCCL",
    "_ProcessGroupWrapper",
    # Availability flags
    "_MPI_AVAILABLE",
    "_NCCL_AVAILABLE",
    "_GLOO_AVAILABLE",
    "_UCC_AVAILABLE",
    "_XCCL_AVAILABLE",
    # Control flag
    "HAS_DISTRIBUTED",
]
