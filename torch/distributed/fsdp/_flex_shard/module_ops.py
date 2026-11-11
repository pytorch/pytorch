# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Operations on a sharded module's buckets, shared by ``FSDPModule`` and
``FlexShardModule``: explicit unshard and reshard, and the settings that their
setters give each bucket."""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import torch
import torch.distributed as dist

from .bucket_runtime import _alloc_storage, _free_storage, _inner_tensors


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .all_gather_layout import AllGatherLayout
    from .bucket_runtime import BucketCommContext, BucketRuntime
    from .bucket_storage import ShardedBucketStorage


class GroupMesh:
    """The 1-D view of a process group that engine placements and buckets use
    as their mesh."""

    ndim = 1
    mesh_dim_names = None

    def __init__(self, group: dist.ProcessGroup | None, device_type: str) -> None:
        # No group: this rank alone (e.g. replicate(), which does not shard)
        self._group = group
        self.device_type = device_type

    @property
    def mesh(self) -> torch.Tensor:
        """The group's ranks, as ``DeviceMesh.mesh``."""
        ranks = (
            [0] if self._group is None else dist.get_process_group_ranks(self._group)
        )
        return torch.tensor(ranks, dtype=torch.int)

    def get_group(self, mesh_dim: Any = None) -> dist.ProcessGroup | None:
        return self._group

    def size(self, mesh_dim: Any = None) -> int:
        return 1 if self._group is None else self._group.size()

    def get_local_rank(self, mesh_dim: Any = None) -> int:
        return 0 if self._group is None else self._group.rank()


# Unshard and reshard #
def reshard(runtimes: Sequence[BucketRuntime]) -> None:
    """Reshard each bucket, releasing a prefetch of it."""
    for runtime in runtimes:
        runtime.context.release_pending_unshard(runtime)
        if runtime.is_unsharded:
            runtime.reshard()


def unshard(runtimes: Sequence[BucketRuntime], async_op: bool) -> list[BucketRuntime]:
    """Start (``async_op``) or finish each bucket's unshard on the current
    stream; return the buckets to wait on (see ``wait_for_unshard``)."""
    started = []
    for runtime in runtimes:
        if runtime.is_unsharded:
            continue
        if async_op:
            runtime.context.prefetch(runtime, on_current_stream=True)
        else:
            runtime.unshard(on_current_stream=True)
        started.append(runtime)
    return started


def wait_for_unshard(runtimes: Sequence[BucketRuntime]) -> None:
    for runtime in runtimes:
        if any(p.bucket is runtime for p in runtime.context.pending_unshards):
            runtime.unshard()


def _sharded_storages(
    runtimes: Sequence[BucketRuntime],
) -> list[tuple[torch.Tensor, int]]:
    """The tensors whose storage holds the buckets' sharded parameters, with
    its size in bytes: each bucket's storage, or for parameters with storage of
    their own, their data, the inner tensors for tensor subclasses."""
    storages = []
    for runtime in runtimes:
        homing = runtime.representation.homing
        if homing is None or not homing.own_storage:
            storage = runtime.bucket_storage
            storages.append((storage._byte_storage, storage._total_bytes))
            continue
        for slot in homing.slots:
            for tensor in _inner_tensors(slot):
                numel = int(tensor.storage_offset()) + tensor.numel()
                storages.append((tensor, numel * tensor.itemsize))
    return storages


def free_sharded_params(runtimes: Sequence[BucketRuntime]) -> None:
    for tensor, _ in _sharded_storages(runtimes):
        _free_storage(tensor)


def restore_sharded_params(runtimes: Sequence[BucketRuntime]) -> None:
    for tensor, nbytes in _sharded_storages(runtimes):
        _alloc_storage(tensor, nbytes)


# Settings #
def get_reshard_after_forward(storage: ShardedBucketStorage | None) -> bool | int:
    if storage is None:
        return True
    if not storage._reshard_after_forward:
        return False
    group = storage._post_forward_group
    return True if group is None else group.size()


def set_requires_gradient_sync(
    storages: Sequence[ShardedBucketStorage], requires_gradient_sync: bool
) -> None:
    for storage in storages:
        storage.set_requires_gradient_sync(requires_gradient_sync)
        storage._requires_all_reduce = requires_gradient_sync


def set_requires_all_reduce(
    storages: Sequence[ShardedBucketStorage], requires_all_reduce: bool
) -> None:
    for storage in storages:
        storage._requires_all_reduce = requires_all_reduce


def set_reshard_after_backward(
    storages: Sequence[ShardedBucketStorage], reshard_after_backward: bool
) -> None:
    for storage in storages:
        storage.set_reshard_after_backward(reshard_after_backward)


def set_all_reduce_hook(
    storages: Sequence[ShardedBucketStorage],
    hook: Callable[[torch.Tensor], None],
    stream: torch.Stream | None,
) -> None:
    if stream is not None and any(storage._replicate_group for storage in storages):
        raise ValueError("stream cannot be set when using native HSDP")
    for storage in storages:
        storage._all_reduce_hook = hook
        if stream is not None:
            storage._all_reduce_hook_stream = stream


def set_force_sum_reduction_for_comms(
    storages: Sequence[ShardedBucketStorage], enable: bool
) -> None:
    for storage in storages:
        storage.set_force_sum_reduction_for_comms(enable)


def set_gradient_divide_factor(
    storages: Sequence[ShardedBucketStorage], factor: float | None
) -> None:
    for storage in storages:
        storage.set_gradient_divide_factor(factor)


def set_keep_unsharded_storage(runtimes: Sequence[BucketRuntime], keep: bool) -> None:
    for runtime in runtimes:
        runtime.keep_persistent_storage = keep
        state = runtime.placement_state
        if state is not None and hasattr(state, "keep_storage"):
            state.keep_storage = keep  # type: ignore[attr-defined]


def set_prefetch(
    runtimes: Sequence[BucketRuntime],
    targets: Sequence[BucketRuntime],
    *,
    forward: bool,
) -> None:
    storages = [target.bucket_storage for target in targets]
    for runtime in runtimes:
        if forward:
            runtime.bucket_storage.set_buckets_to_forward_prefetch(storages)
        else:
            runtime.bucket_storage.set_buckets_to_backward_prefetch(storages)


def set_reduce_scatter_input_fn(
    storages: Sequence[ShardedBucketStorage], fn: Callable | None
) -> None:
    """``None`` restores the placement's own copy-in."""
    for storage in storages:
        storage._reduce_scatter_input_fn = fn


def set_reduce_scatter_unused_params(
    storages: Sequence[ShardedBucketStorage], enable: bool
) -> None:
    for storage in storages:
        storage.set_reduce_scatter_unused_params(enable)


def set_unshard_async_op(
    storages: Sequence[ShardedBucketStorage], async_op: bool
) -> None:
    for storage in storages:
        storage._unshard_async_op = async_op


def set_unshard_in_backward(
    storages: Sequence[ShardedBucketStorage], unshard_in_backward: bool
) -> None:
    for storage in storages:
        storage._unshard_in_backward = unshard_in_backward


def set_separate_reduce_scatter_group(
    storages: Sequence[ShardedBucketStorage],
    enable: bool,
    reduce_meshes: dict | None = None,
) -> None:
    """Run the buckets' reduce-scatters on a process group of their own over
    their mesh's ranks, one per distinct set of ranks (``reduce_meshes`` keeps
    the ones made so far), or with ``enable=False`` on their mesh's group (see
    ``ShardedBucketStorage.set_separate_reduce_scatter_group``)."""
    reduce_meshes = {} if reduce_meshes is None else reduce_meshes
    for storage in storages:
        storage.set_separate_reduce_scatter_group(enable, reduce_meshes)


def set_max_pending_reduce_grads(context: BucketCommContext, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            "max_pending_reduce_grads must be a non-negative int, "
            f"got {type(value).__name__}."
        )
    if value < 0:
        raise ValueError(f"max_pending_reduce_grads must be non-negative, got {value}.")
    context.max_pending_reduce_grads = value


# Comms #
def _check_single_bucket(buckets: Sequence[Any], method: str) -> None:
    if len(buckets) > 1:
        raise ValueError(
            f"{method} is not supported with multiple param groups (from "
            "per-param mesh via shard_placement_fn). The custom comm would be "
            "ambiguous across groups with different meshes."
        )


def _all_gather_comm_type(runtime: BucketRuntime) -> type:
    """The type of the bucket's all-gather comm, ``DefaultAllGather`` if unset."""
    from .comms import DefaultAllGather

    comm = runtime.bucket_storage._all_gather_comm
    return DefaultAllGather if comm is None else type(comm)


def _layout(runtime: BucketRuntime) -> AllGatherLayout:
    return runtime.placement_state.layout  # type: ignore[union-attr]


def _check_layout_comm(layout: Any, comm_type: type) -> None:
    if layout.comm_type is not None and not issubclass(comm_type, layout.comm_type):
        raise ValueError(
            f"all-gather layout {type(layout).__name__} requires a "
            f"{layout.comm_type.__name__} comm, not {comm_type.__name__}; "
            "install one with set_custom_all_gather() before the layout"
        )


def _check_no_custom_comms(runtime: BucketRuntime, method: str) -> None:
    """FSDP's comm settings replace only comms that declare it (``Comm.
    _replaceable_by_fsdp``), as FSDP's own do, not custom ones."""
    storage = runtime.bucket_storage
    for comm in (storage._all_gather_comm, storage._reduce_scatter_comm):
        # Looked up on the type: a mock of the interface is a custom comm
        if comm is not None and not getattr(type(comm), "_replaceable_by_fsdp", False):
            raise AssertionError(
                f"cannot call {method} when the comm is custom: {type(comm).__name__}"
            )


def set_custom_all_gather(runtimes: Sequence[BucketRuntime], comm: Any) -> None:
    _check_single_bucket(runtimes, "set_custom_all_gather")
    for runtime in runtimes:
        _check_layout_comm(_layout(runtime), type(comm))
    for runtime in runtimes:
        runtime.bucket_storage._all_gather_comm = comm


def set_custom_reduce_scatter(
    storages: Sequence[ShardedBucketStorage], comm: Any
) -> None:
    _check_single_bucket(storages, "set_custom_reduce_scatter")
    for storage in storages:
        storage._reduce_scatter_comm = comm


def set_allocate_memory_from_process_group(
    runtimes: Sequence[BucketRuntime], enable: bool
) -> None:
    from .comms import (
        DefaultAllGather,
        ProcessGroupAllocAllGather,
        ProcessGroupAllocReduceScatter,
    )

    comm_type = ProcessGroupAllocAllGather if enable else DefaultAllGather
    for runtime in runtimes:
        _check_layout_comm(_layout(runtime), comm_type)
        _check_no_custom_comms(runtime, "set_allocate_memory_from_process_group()")
    for runtime in runtimes:
        storage = runtime.bucket_storage
        group = storage._mesh.get_group()
        reduce_group = storage.reduce_mesh.get_group()
        storage._all_gather_comm = ProcessGroupAllocAllGather(group) if enable else None
        storage._reduce_scatter_comm = (
            ProcessGroupAllocReduceScatter(reduce_group) if enable else None
        )


def set_symm_mem(runtimes: Sequence[BucketRuntime], backend: Literal["NCCL"]) -> None:
    from .comms import SymmMemAllGather, SymmMemReduceScatter

    for runtime in runtimes:
        _check_layout_comm(_layout(runtime), SymmMemAllGather)
        _check_no_custom_comms(runtime, "set_symm_mem()")
    for runtime in runtimes:
        storage = runtime.bucket_storage
        storage._all_gather_comm = SymmMemAllGather(storage._mesh.get_group(), backend)
        if storage.gradient_reduction.force_sum:
            # As of NCCL 2.29.3, NCCL symmetric reduce-scatter only supports SUM
            storage._reduce_scatter_comm = SymmMemReduceScatter(
                storage.reduce_mesh.get_group(), backend
            )


def set_all_gather_layout(
    runtimes: Sequence[BucketRuntime], layout: AllGatherLayout
) -> None:
    for runtime in runtimes:
        _check_layout_comm(layout, _all_gather_comm_type(runtime))
        all_gather = runtime.placement_state
        if layout is _layout(runtime):
            continue
        if runtime.is_unsharded or any(
            pending.bucket is runtime for pending in runtime.context.pending_unshards
        ):
            raise RuntimeError(
                "cannot replace the all-gather layout while an all-gather is "
                "pending or parameters are unsharded; reshard, or call "
                "reset_iter_state() after a failed unshard"
            )
        if any(p.outputs_view_layout_buffers for p in all_gather.params):  # type: ignore[union-attr]
            raise RuntimeError(
                "cannot replace the all-gather layout after parameters adopted "
                "outputs from the installed custom layout"
            )
    for runtime in runtimes:
        all_gather = runtime.placement_state
        # Outputs of unevenly sharded Shard(i), i > 0, parameters keep the
        # padding inside their rows only under a custom layout (see
        # ParamAllGather), so they carry over only between layouts that agree
        if (
            layout._rank_major_output_only != all_gather.layout._rank_major_output_only  # type: ignore[union-attr]
            and any(
                p.drops_row_padding and p.all_gather_outputs
                for p in all_gather.params  # type: ignore[union-attr]
            )
        ):
            runtime.unsharded_params = None
            runtime.persistent_buffers = []
            runtime.persistent_buffer_nbytes = []
            all_gather.reset()  # type: ignore[union-attr]
        all_gather.layout = layout  # type: ignore[union-attr]
