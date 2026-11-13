"""
Manages process groups for distributed compilation in TorchDynamo.

This module handles the initialization and management of process groups used for
distributed compilation. Key features:

- Lazy initialization of compilation process groups
- Only creates groups when distributed mode is enabled and available
- Integrates with compiler_collectives configuration setting
- Provides a single global process group for compilation coordination

The process group is created only when needed and if the distributed environment
is properly initialized, making it safe to import and use this module even in
non-distributed scenarios.
"""

import torch
import torch.distributed as dist

from . import config


_COMPILE_PG: dist.ProcessGroup | None = None
_GUARD_PG: dist.ProcessGroup | None = None
_COMPILE_SYNC_PG: dist.ProcessGroup | None = None


def get_compile_pg() -> dist.ProcessGroup | None:
    if (
        config.enable_compiler_collectives
        and dist.is_available()
        and dist.is_initialized()
    ):
        global _COMPILE_PG
        if _COMPILE_PG is None:
            compile_pg = dist.distributed_c10d._new_group_with_tag(
                pg_tag="pt2_compile_pg"
            )
            if compile_pg == dist.GroupMember.NON_GROUP_MEMBER:
                raise AssertionError("Compiler process group must include all ranks")
            _COMPILE_PG = compile_pg
        return _COMPILE_PG

    return None


# NB: Unlike get_compile_pg, this is only called when guard collectives were
# explicitly requested
def get_guard_pg() -> dist.ProcessGroup | None:
    if dist.is_available() and dist.is_initialized():
        global _GUARD_PG
        if _GUARD_PG is None:
            guard_pg = dist.distributed_c10d._new_group_with_tag(pg_tag="pt2_guard_pg")
            if guard_pg == dist.GroupMember.NON_GROUP_MEMBER:
                raise AssertionError("Guard process group must include all ranks")
            _GUARD_PG = guard_pg
        return _GUARD_PG

    return None


# NB: Like get_guard_pg, this is only called when the caller explicitly asked for
# compile time synchronization.
def get_compile_sync_pg() -> dist.ProcessGroup | None:
    """
    Process group for collectives issued from inside the compiler itself, e.g. the
    partitioner's cross rank decision sync.

    These must not share a process group with the model's runtime collectives: ranks
    reach a given compile at different times, so a rank that has already resumed
    execution can otherwise match one of its runtime ops against another rank's
    compile time op. We thus require gloo as the backend. Gloo keeps the traffic off
    the accelerator as well, so a compile time collective can't interleave with
    an in flight NCCL op.
    """
    if dist.is_available() and dist.is_initialized():
        global _COMPILE_SYNC_PG
        if _COMPILE_SYNC_PG is None:
            if not dist.is_gloo_available():
                # No fallback: per above, the accelerator backend would let a compile
                # time collective match a runtime one.
                raise RuntimeError(
                    "Compile time cross rank synchronization requires the gloo "
                    "backend, which this build of PyTorch does not have. Either "
                    "build with gloo, or turn off "
                    "torch._functorch.config._sync_decision_cross_ranks and "
                    "torch._functorch.config._sync_cache_decision_cross_ranks."
                )
            # Left to itself, _new_group_with_tag picks the global default_pg_timeout
            # constant rather than the timeout init_process_group was configured with.
            default_pg = dist.distributed_c10d._get_default_group()
            coll_device = dist.distributed_c10d._get_object_coll_device(default_pg)
            default_backend = default_pg._get_backend(torch.device(coll_device))
            compile_sync_pg = dist.distributed_c10d._new_group_with_tag(
                backend="gloo",
                timeout=default_backend.options._timeout,
                pg_tag="pt2_compile_sync_pg",
            )
            if compile_sync_pg == dist.GroupMember.NON_GROUP_MEMBER:
                raise AssertionError(
                    "Compile sync process group must include all ranks"
                )
            _COMPILE_SYNC_PG = compile_sync_pg
        return _COMPILE_SYNC_PG

    return None
