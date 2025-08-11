# mypy: disable-error-code="assignment"
# noqa: F401
# NB: Disable assignment errors as all of the stub alternative imports
# reassign over symbols
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

from typing import TYPE_CHECKING


# Single minimal try-catch block to import the C extension
try:
    import torch._C._distributed_c10d as _C
except (ImportError, AttributeError):
    _C = None

# Set HAS_DISTRIBUTED based on whether C extension is available
HAS_DISTRIBUTED = _C is not None

_MPI_AVAILABLE = False
_NCCL_AVAILABLE = False
_GLOO_AVAILABLE = False
_UCC_AVAILABLE = False
_XCCL_AVAILABLE = False

if HAS_DISTRIBUTED or TYPE_CHECKING:
    # NB: This list has to be spelled out because the _C module doesn't have
    # __all__
    from torch._C._distributed_c10d import (
        _allow_inflight_collective_as_graph_input,
        _broadcast_coalesced,
        _compute_bucket_assignment_by_size,
        _ControlCollectives,
        _current_process_group,
        _DEFAULT_FIRST_BUCKET_BYTES,
        _DEFAULT_PG_TIMEOUT,
        _DistributedBackendOptions,
        _make_nccl_premul_sum,
        _register_builtin_comm_hook,
        _register_comm_hook,
        _register_process_group,
        _register_work,
        _resolve_process_group,
        _set_allow_inflight_collective_as_graph_input,
        _set_global_rank,
        _set_process_group,
        _StoreCollectives,
        _test_python_store,
        _unregister_all_process_groups,
        _unregister_process_group,
        _verify_params_across_processes,
        _WorkerServer,
        AllgatherOptions,
        AllreduceCoalescedOptions,
        AllreduceOptions,
        AllToAllOptions,
        Backend,
        BarrierOptions,
        BroadcastOptions,
        BuiltinCommHookType,
        DebugLevel,
        FakeProcessGroup,
        FakeWork,
        FileStore,
        GatherOptions,
        get_debug_level,
        GradBucket,
        Logger,
        PrefixStore,
        ProcessGroup,
        ReduceOp,
        ReduceOptions,
        Reducer,
        ReduceScatterOptions,
        ScatterOptions,
        set_debug_level,
        set_debug_level_from_env,
        Store,
        TCPStore,
        Work,
    )

    # These identifiers aren't always available on all builds
    try:
        from torch._C._distributed_c10d import HashStore
    except ImportError:
        from torch.distributed._C_stubs import HashStore

    try:
        from torch._C._distributed_c10d import (
            _is_nvshmem_available,
            _nvshmemx_cumodule_init,
            _SymmetricMemory,
        )
    except ImportError:
        # Fall back to stub versions if C extension doesn't have these
        from torch.distributed._C_stubs import (
            _is_nvshmem_available,
            _nvshmemx_cumodule_init,
            _SymmetricMemory,
        )

    try:
        from torch._C._distributed_c10d import ProcessGroupMPI

        _MPI_AVAILABLE = True
    except ImportError:
        from torch.distributed._C_stubs import ProcessGroupMPI

    try:
        from torch._C._distributed_c10d import (
            _DEFAULT_PG_NCCL_TIMEOUT,
            _dump_nccl_trace,
            _dump_nccl_trace_json,
            _hash_tensors,
            ProcessGroupNCCL,
        )

        _NCCL_AVAILABLE = True
    except ImportError:
        from torch.distributed._C_stubs import (
            _DEFAULT_PG_NCCL_TIMEOUT,
            _hash_tensors,
            ProcessGroupNCCL,
        )

    try:
        from torch._C._distributed_c10d import _ProcessGroupWrapper, ProcessGroupGloo

        _GLOO_AVAILABLE = True
    except ImportError:
        from torch.distributed._C_stubs import _ProcessGroupWrapper, ProcessGroupGloo

    try:
        from torch._C._distributed_c10d import ProcessGroupUCC

        _UCC_AVAILABLE = True
    except ImportError:
        from torch.distributed._C_stubs import ProcessGroupUCC

    try:
        from torch._C._distributed_c10d import ProcessGroupXCCL

        _XCCL_AVAILABLE = True
    except ImportError:
        from torch.distributed._C_stubs import ProcessGroupXCCL

else:
    from torch.distributed._C_stubs import (
        _allow_inflight_collective_as_graph_input,
        _broadcast_coalesced,
        _compute_bucket_assignment_by_size,
        _ControlCollectives,
        _current_process_group,
        _DEFAULT_FIRST_BUCKET_BYTES,
        _DEFAULT_PG_NCCL_TIMEOUT,
        _DEFAULT_PG_TIMEOUT,
        _DistributedBackendOptions,
        _hash_tensors,
        _is_nvshmem_available,
        _make_nccl_premul_sum,
        _nvshmemx_cumodule_init,
        _ProcessGroupWrapper,
        _register_builtin_comm_hook,
        _register_comm_hook,
        _register_process_group,
        _register_work,
        _resolve_process_group,
        _set_allow_inflight_collective_as_graph_input,
        _set_global_rank,
        _set_process_group,
        _StoreCollectives,
        _SymmetricMemory,
        _test_python_store,
        _unregister_all_process_groups,
        _unregister_process_group,
        _verify_params_across_processes,
        _WorkerServer,
        AllgatherOptions,
        AllreduceCoalescedOptions,
        AllreduceOptions,
        AllToAllOptions,
        Backend,
        BarrierOptions,
        BroadcastOptions,
        BuiltinCommHookType,
        DebugLevel,
        FakeProcessGroup,
        FakeWork,
        FileStore,
        GatherOptions,
        get_debug_level,
        GradBucket,
        HashStore,
        Logger,
        PrefixStore,
        ProcessGroup,
        ProcessGroupGloo,
        ProcessGroupMPI,
        ProcessGroupNCCL,
        ProcessGroupUCC,
        ProcessGroupXCCL,
        ReduceOp,
        ReduceOptions,
        Reducer,
        ReduceScatterOptions,
        ScatterOptions,
        set_debug_level,
        set_debug_level_from_env,
        Store,
        TCPStore,
        Work,
    )

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
    "_hash_tensors",
    "_set_global_rank",
    "_dump_nccl_trace",
    "_dump_nccl_trace_json",
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
