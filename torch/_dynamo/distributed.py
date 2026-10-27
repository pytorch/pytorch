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


def _is_fake_world() -> bool:
    """Whether every backend of the default group is fake, so no rank has a real peer
    to rendezvous with."""
    config = dist.BackendConfig(dist.get_backend_config())
    return all(
        backend == dist.Backend.FAKE
        for backend in config.get_device_backend_map().values()
    )


def _compile_sync_backend() -> str | None:
    """The backend get_compile_sync_pg builds its group on, or None when this build
    cannot build it (a real world without gloo)."""
    if _is_fake_world():
        return dist.Backend.FAKE
    if dist.is_gloo_available():
        return dist.Backend.GLOO
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

    The exception is a fake default group, where one real rank stands in for the
    whole world and has no peers to rendezvous with over gloo. Its sync group is
    fake as well, so every compile time collective returns the local value: fake
    allgather copies the input into every output slot and MIN/MAX all_reduce leave
    it unchanged, which makes each decision the local one.
    """
    if dist.is_available() and dist.is_initialized():
        global _COMPILE_SYNC_PG
        if _COMPILE_SYNC_PG is None:
            backend = _compile_sync_backend()
            if backend is None:
                # No fallback: per above, the accelerator backend would let a
                # compile time collective match a runtime one.
                raise RuntimeError(
                    "Compile time cross rank synchronization requires the gloo "
                    "backend, which this build of PyTorch does not have. Either "
                    "build with gloo, or turn off "
                    "torch._functorch.config._sync_decision_cross_ranks and "
                    "torch._functorch.config._sync_cache_decision_cross_ranks."
                )
            # Left to itself, _new_group_with_tag picks the global
            # default_pg_timeout constant rather than the timeout
            # init_process_group was configured with.
            default_pg = dist.distributed_c10d._get_default_group()
            coll_device = dist.distributed_c10d._get_object_coll_device(default_pg)
            # A fake backend built without options has none, so fall back to another
            # device's backend (cuda's in cpu:fake,cuda:nccl); none has options only
            # in a fake world.
            timeout = None
            for device in (torch.device(coll_device), *default_pg._device_types):
                options = default_pg._get_backend(device).options
                if options is not None:
                    timeout = options._timeout
                    break
            compile_sync_pg = dist.distributed_c10d._new_group_with_tag(
                backend=backend, timeout=timeout, pg_tag="pt2_compile_sync_pg"
            )
            if compile_sync_pg == dist.GroupMember.NON_GROUP_MEMBER:
                raise AssertionError(
                    "Compile sync process group must include all ranks"
                )
            _COMPILE_SYNC_PG = compile_sync_pg
        return _COMPILE_SYNC_PG

    return None
