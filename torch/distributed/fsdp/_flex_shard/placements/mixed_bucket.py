# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist

from ..placement_contract import (
    BucketParamStorageLayout,
    BucketStorageLayout,
    Placement,
    PlacementPreparedReduceGrad,
    PlacementPreparedUnshard,
    PlacementReduceGradResult,
    PlacementUnshardResult,
)
from ..utils import (
    _record_comm_if_eager,
    _record_copy_in_if_eager,
    _record_copy_out_if_eager,
    _record_function_if_eager,
)
from .block_shard import (
    _grad_reduce_dtype,
    _unsharded_dtype,
    BlockShard,
    BucketedBlockShard,
)
from .blockwise_quant_bucketed_block_shard import (
    _align_up,
    _VEC_ALIGN_BYTES,
    BlockwiseQuantBucketedBlockShard,
)
from .owned import BucketedOwned, BucketedOwnedSegmentSpec
from .owner_rows_transport import (
    gather_owner_rows,
    owner_rows_transport_buffers,
    owner_rows_transport_plan,
    OwnerRowsTransportBuffers,
    OwnerRowsTransportPlan,
    reduce_owner_rows,
)
from .shard import Shard
from .utils import foreach_copy_, reduce_scatter_grads, reduce_scatter_grads_uneven


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..bucket_storage import ParamInfo, PlacementFn
    from ..placement_contract import GradientReduction


@dataclass(frozen=True)
class _PlacementGroup:
    placement: Placement
    key: object
    indices: list[int]
    tensors: list[torch.Tensor]
    infos: list[ParamInfo]


@dataclass(frozen=True)
class _MixedUnshardGroupLayout:
    offset: int
    numel: int
    owner_rank: int | None


@dataclass(frozen=True)
class _MixedUnshardLayout:
    groups: tuple[_MixedUnshardGroupLayout, ...]
    dtype: torch.dtype
    requires_dtype_reinterpretation: bool
    row_numel: int
    padding_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _MixedUnshardGroupState:
    prepared: PlacementPreparedUnshard
    indices: tuple[int, ...]
    offset: int
    numel: int


@dataclass(frozen=True)
class _MixedUnshardState:
    groups: list[_MixedUnshardGroupState]
    world_size: int
    row_numel: int
    requires_dtype_reinterpretation: bool
    pg: Any
    debug_fqn: str | None


@dataclass(frozen=True)
class _OwnerRowsGather:
    """A gather through equal chunks (see owner_rows_transport): the plan, the
    mesh ranks' hosts, the transport buffers, this mesh rank, and its params
    with their starts in the gathered buffer."""

    plan: OwnerRowsTransportPlan
    hosts: tuple[int, ...]
    buffers: OwnerRowsTransportBuffers
    rank: int
    sources: list[tuple[torch.Tensor, int]]


@dataclass(frozen=True)
class _OwnerRowsUnshardState:
    """Unshard state of a bucket whose groups each live whole on one owner,
    whose params view the gathered rows (see
    MixedBucketPlacement._prepare_owner_rows_unshard). ``row_numels`` are the
    rows' lengths, back to back in the gathered buffer. With ``transport``,
    the rows go through equal chunks and the buffer is padded to whole chunks;
    without, they're gathered at their own lengths."""

    groups: list[_MixedUnshardGroupState]
    world_size: int
    row_numels: tuple[int, ...]
    pg: Any
    debug_fqn: str | None
    transport: _OwnerRowsGather | None = None


@dataclass(frozen=True)
class _OwnerRowsReduce:
    """A reduce through equal chunks (see owner_rows_transport): the plan, the
    mesh ranks' hosts, the transport buffers, this mesh rank, and the dtype
    its row comes back in: its sharded grads' dtype, so the cast to it happens
    on the way."""

    plan: OwnerRowsTransportPlan
    hosts: tuple[int, ...]
    buffers: OwnerRowsTransportBuffers
    rank: int
    recv_dtype: torch.dtype


@dataclass(frozen=True)
class _MixedReduceGradGroupState:
    prepared: PlacementPreparedReduceGrad
    indices: list[int]
    offset: int
    numel: int


@dataclass(frozen=True)
class _MixedReduceGradState:
    groups: list[_MixedReduceGradGroupState]
    world_size: int
    row_numel: int
    pg: Any
    debug_fqn: str | None
    # Owner rows only: each rank's row length, back to back in the send buffer
    # (see _owner_rows_reduce_state); row_numel is then this rank's. With
    # transport, the send buffer is padded to whole chunks.
    row_numels: tuple[int, ...] | None = None
    transport: _OwnerRowsReduce | None = None


class _MixedBucketMember:
    _mixed_bucket: MixedBucketPlacement
    _collective_placement: Placement

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _MixedBucketMember):
            other = other._collective_placement
        return self._collective_placement == other

    def __hash__(self) -> int:
        return hash(self._collective_placement)

    def bucket_compatibility_key(self) -> object:
        return self._mixed_bucket

    def bucket_storage_layout(
        self,
        named_params: list[tuple[str, torch.nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        mesh: DeviceMesh,
    ) -> BucketStorageLayout | None:
        return self._mixed_bucket.bucket_storage_layout(
            named_params,
            param_placements,
            mesh,
        )

    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        return self._mixed_bucket.prepare_unshard_bucket(
            tensors,
            infos,
            mesh,
            debug_fqn,
        )

    def run_prepared_unshard(
        self, prepared: PlacementPreparedUnshard, *, async_op: bool = False
    ) -> dist.Work | None:
        return self._mixed_bucket.run_prepared_unshard(prepared, async_op=async_op)

    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        return self._mixed_bucket.finish_prepared_unshard(prepared)

    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad:
        return self._mixed_bucket.prepare_reduce_grad(
            tensors,
            infos,
            mesh,
            debug_fqn,
        )

    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
        reduction: GradientReduction,
    ) -> PlacementReduceGradResult:
        return self._mixed_bucket.reduce_prepared_grad(prepared, reduction)

    def reduces_grads_in_param_layout(
        self, infos: list[ParamInfo], world_size: int
    ) -> bool:
        return self._mixed_bucket.reduces_grads_in_param_layout(infos, world_size)


class _MixedShard0(_MixedBucketMember, Shard):
    """Optimizer-visible Shard(0) member of one mixed bucket placement."""

    __eq__ = _MixedBucketMember.__eq__
    __hash__ = _MixedBucketMember.__hash__

    def __init__(self, mixed_bucket: MixedBucketPlacement) -> None:
        Shard.__init__(self, 0)
        self._mixed_bucket = mixed_bucket
        self._collective_placement = mixed_bucket._shard


class _MixedBlockShard(_MixedBucketMember, BlockShard):
    """Optimizer-visible per-parameter BlockShard mixed-bucket member."""

    __eq__ = _MixedBucketMember.__eq__
    __hash__ = _MixedBucketMember.__hash__

    def __init__(
        self,
        mixed_bucket: MixedBucketPlacement,
        blocks_per_rank: tuple[int, ...],
        dim: int,
    ) -> None:
        BlockShard.__init__(self, blocks_per_rank=blocks_per_rank, dim=dim)
        self._mixed_bucket = mixed_bucket
        self._collective_placement = BlockShard(
            blocks_per_rank=blocks_per_rank,
            dim=dim,
        )


class _MixedBucketedOwned(_MixedBucketMember, BucketedOwned):
    """Optimizer-visible BucketedOwned member of one mixed bucket placement."""

    __eq__ = _MixedBucketMember.__eq__
    __hash__ = _MixedBucketMember.__hash__

    def __init__(
        self,
        mixed_bucket: MixedBucketPlacement,
        segments_by_fqn: dict[str, list[Any]],
    ) -> None:
        BucketedOwned.__init__(
            self,
            segments_by_fqn,
            view_kind="full_param",
        )
        self._mixed_bucket = mixed_bucket
        self._collective_placement = mixed_bucket._bucketed_owned


class _MixedBlockwiseQuantMember(
    _MixedBucketMember,
    BlockwiseQuantBucketedBlockShard,
):
    """Optimizer-visible blockwise quantized member of one mixed bucket placement."""

    __eq__ = _MixedBucketMember.__eq__
    __hash__ = _MixedBucketMember.__hash__


# Each member type's constructor takes the mixed bucket and the collective
# placement it copies. Quantized placements outside core (e.g. FP8 and MXFP8
# in torchao or the flex_shard repo) register theirs.
_MIXED_BLOCKWISE_QUANT_MEMBER_TYPES: dict[
    type[BlockwiseQuantBucketedBlockShard],
    Callable[[MixedBucketPlacement, Any], _MixedBlockwiseQuantMember],
] = {}


def register_mixed_blockwise_quant_member(
    placement_type: type[BlockwiseQuantBucketedBlockShard],
    member_type: Callable[[MixedBucketPlacement, Any], _MixedBlockwiseQuantMember],
) -> None:
    """Let mixed buckets hold ``placement_type`` members, built by ``member_type``."""
    _MIXED_BLOCKWISE_QUANT_MEMBER_TYPES[placement_type] = member_type


class MixedBucketPlacement(Placement):
    """One collective for the built-in mixed-bucket placement groups."""

    def __init__(self, segments_by_fqn: dict[str, list[Any]]) -> None:
        self._shard = Shard(0)
        self._bucketed_owned = BucketedOwned(
            segments_by_fqn,
            view_kind="full_param",
        )
        self.shard0 = _MixedShard0(self)
        self.bucketed_owned = _MixedBucketedOwned(
            self,
            segments_by_fqn,
        )
        self._block_shard_members_by_config: dict[
            tuple[int, tuple[int, ...]], _MixedBlockShard
        ] = {}
        self._blockwise_quant_member: _MixedBlockwiseQuantMember | None = None

    def block_shard(
        self,
        *,
        blocks_per_rank: tuple[int, ...],
        dim: int = 0,
    ) -> _MixedBlockShard:
        key = (dim, blocks_per_rank)
        member = self._block_shard_members_by_config.get(key)
        if member is None:
            member = _MixedBlockShard(self, blocks_per_rank, dim)
            self._block_shard_members_by_config[key] = member
        return member

    def blockwise_quant_bucketed_block_shard(
        self,
        placement: BlockwiseQuantBucketedBlockShard,
    ) -> BlockwiseQuantBucketedBlockShard:
        member = self._blockwise_quant_member
        if member is None:
            member_type = _MIXED_BLOCKWISE_QUANT_MEMBER_TYPES.get(type(placement))
            if member_type is None:
                raise TypeError(
                    "Mixed FlexShard buckets do not support blockwise quantized "
                    f"placement {type(placement).__name__}."
                )
            member = member_type(self, placement)
            self._blockwise_quant_member = member
        elif member._collective_placement is not placement:
            raise NotImplementedError(
                "A mixed bucket supports one blockwise quantized placement instance."
            )
        return member

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return hash(id(self))

    def compute_local_shape(
        self,
        global_shape: torch.Size,
        rank: int,
        world_size: int,
    ) -> torch.Size:
        raise NotImplementedError("Mixed bucket placement is bucket-scoped.")

    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        raise NotImplementedError("Mixed bucket placement is bucket-scoped.")

    def bucket_storage_layout(
        self,
        named_params: list[tuple[str, torch.nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        mesh: DeviceMesh,
    ) -> BucketStorageLayout | None:
        rank = mesh.get_local_rank()
        world_size = mesh.size()
        local_params: list[tuple[str, torch.nn.Parameter, Placement]] = []
        for fqn, param in named_params:
            (placement,) = param_placements[fqn]
            if not isinstance(placement, _MixedBucketMember):
                raise TypeError(
                    "MixedBucketPlacement requires mixed member "
                    f"placements, but {fqn!r} uses {type(placement).__name__}."
                )
            if placement._mixed_bucket is not self:
                raise ValueError(
                    "MixedBucketPlacement cannot mix member "
                    "placements from different bucket instances."
                )
            local_params.append(
                (
                    fqn,
                    param,
                    _collective_placement(
                        placement,
                        fqn,
                        param.shape,
                        world_size,
                    ),
                )
            )
        owner_rows = self._owner_rows_storage_layout(local_params, rank, world_size)
        if owner_rows is not None:
            return owner_rows

        param_layouts: dict[str, BucketParamStorageLayout] = {}
        byte_offset = 0
        for local_group in _group_local_params_by_placement(local_params):
            collective_placement = local_group[0][2]
            group_named_params = [(fqn, param) for fqn, param, _ in local_group]
            group_storage_layout = collective_placement.bucket_storage_layout(
                group_named_params,
                {fqn: (collective_placement,) for fqn, _ in group_named_params},
                mesh,
            )
            if group_storage_layout is not None:
                for fqn, layout in group_storage_layout.param_layouts.items():
                    param_layouts[fqn] = BucketParamStorageLayout(
                        local_shape=layout.local_shape,
                        local_numel=layout.local_numel,
                        byte_offset=byte_offset + layout.byte_offset,
                        storage_nbytes=layout.storage_nbytes,
                        bucket_layout=layout.bucket_layout,
                    )
                byte_offset += group_storage_layout.total_bytes
                continue

            for fqn, param, placement in local_group:
                local_layout = placement.local_storage_layout(
                    param.shape,
                    param.dtype,
                    rank,
                    world_size,
                )
                offset = byte_offset if local_layout.storage_nbytes > 0 else 0
                param_layouts[fqn] = BucketParamStorageLayout(
                    local_shape=local_layout.local_shape,
                    local_numel=local_layout.local_numel,
                    byte_offset=offset,
                    storage_nbytes=local_layout.storage_nbytes,
                )
                byte_offset += local_layout.storage_nbytes

        return BucketStorageLayout(
            param_layouts=param_layouts,
            total_bytes=byte_offset,
        )

    def _owner_rows_storage_layout(
        self,
        local_params: list[tuple[str, torch.nn.Parameter, Placement]],
        rank: int,
        world_size: int,
    ) -> BucketStorageLayout | None:
        """Lay out this rank's params as its row of the gathered buffer, at the
        row's own length, for a bucket whose params each live whole on one
        owner, in one dtype (see _prepare_owner_rows_unshard), so the row is the
        all-gather's input as is. None for any other bucket."""
        dtype = local_params[0][1].dtype
        if any(
            _one_hot_block_owner(placement, world_size) is None or param.dtype != dtype
            for _, param, placement in local_params
        ):
            return None
        groups = _group_local_params_by_placement(local_params)
        group_numels = [sum(param.numel() for _, param, _ in group) for group in groups]
        offsets, owner_ranks, _, _ = _plan_group_offsets(
            [group[0][2] for group in groups],
            group_numels,
            world_size,
        )
        row_numels = _owner_row_numels(owner_ranks, offsets, group_numels, world_size)
        param_layouts: dict[str, BucketParamStorageLayout] = {}
        for group, offset, owner_rank in zip(groups, offsets, owner_ranks, strict=True):
            for fqn, param, placement in group:
                local_layout = placement.local_storage_layout(
                    param.shape, param.dtype, rank, world_size
                )
                param_layouts[fqn] = BucketParamStorageLayout(
                    local_shape=local_layout.local_shape,
                    local_numel=local_layout.local_numel,
                    byte_offset=offset * dtype.itemsize if owner_rank == rank else 0,
                    storage_nbytes=local_layout.storage_nbytes,
                )
                offset += param.numel()
        return BucketStorageLayout(
            param_layouts=param_layouts,
            total_bytes=row_numels[rank] * dtype.itemsize,
        )

    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        rank = mesh.get_local_rank()
        world_size = mesh.size()
        groups = _group_tensors_by_placement(
            tensors,
            infos,
            world_size,
            lambda info: info.unsharded_dtype,
        )
        _validate_mixed_groups(groups)
        owner_rows = self._prepare_owner_rows_unshard(groups, mesh, debug_fqn)
        if owner_rows is not None:
            return owner_rows

        prepared_groups: list[tuple[PlacementPreparedUnshard, tuple[int, ...]]] = []
        group_buffers: list[torch.Tensor] = []
        device: torch.device | None = None

        with _record_copy_in_if_eager():
            for group in groups:
                prepared = _prepare_mixed_unshard_group(
                    group.placement,
                    group.tensors,
                    group.infos,
                    mesh,
                    debug_fqn,
                )
                send = prepared.buffers[0]
                if device is None:
                    device = send.device
                elif send.device != device:
                    raise ValueError(
                        "Mixed FlexShard unshard requires one device per bucket, "
                        f"but got {device} and {send.device}."
                    )
                prepared_groups.append((prepared, tuple(group.indices)))
                group_buffers.extend(prepared.buffers)

            if device is None:
                raise AssertionError("Expected at least one mixed bucket group.")
            layout = _build_mixed_unshard_layout(prepared_groups, world_size)
            group_states = [
                _MixedUnshardGroupState(
                    prepared=group_prepared,
                    indices=indices,
                    offset=group_layout.offset,
                    numel=group_layout.numel,
                )
                for (group_prepared, indices), group_layout in zip(
                    prepared_groups,
                    layout.groups,
                    strict=True,
                )
            ]
            send = torch.empty(layout.row_numel, dtype=layout.dtype, device=device)
            _zero_padding(send, layout.padding_ranges)
            for group_state, group_layout in zip(
                group_states,
                layout.groups,
                strict=True,
            ):
                if (
                    group_layout.owner_rank is not None
                    and group_layout.owner_rank != rank
                ):
                    continue
                group_send = group_state.prepared.buffers[0]
                send.narrow(0, group_state.offset, group_state.numel).copy_(
                    group_send.contiguous().view(torch.uint8).reshape(-1)
                    if layout.requires_dtype_reinterpretation
                    else group_send
                )
            gathered = torch.empty(
                world_size * layout.row_numel,
                dtype=layout.dtype,
                device=device,
            )

        return PlacementPreparedUnshard(
            placement=self,
            buffers=[send, gathered, *group_buffers],
            placement_state=_MixedUnshardState(
                groups=group_states,
                world_size=world_size,
                row_numel=layout.row_numel,
                requires_dtype_reinterpretation=(
                    layout.requires_dtype_reinterpretation
                ),
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
            ),
        )

    def run_prepared_unshard(
        self, prepared: PlacementPreparedUnshard, *, async_op: bool = False
    ) -> dist.Work | None:
        if isinstance(prepared.placement_state, _OwnerRowsUnshardState):
            return self._run_owner_rows_unshard(
                prepared, prepared.placement_state, async_op=async_op
            )
        if not isinstance(prepared.placement_state, _MixedUnshardState):
            raise AssertionError(
                "Expected _MixedUnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        with _record_comm_if_eager(
            "FlexShard::mixed_all_gather",
            prepared.placement_state.debug_fqn,
        ):
            return dist.all_gather_into_tensor(
                output_tensor=prepared.buffers[1],
                input_tensor=prepared.buffers[0],
                group=prepared.placement_state.pg,
                async_op=async_op,
            )

    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        if isinstance(prepared.placement_state, _OwnerRowsUnshardState):
            return self._finish_owner_rows_unshard(prepared, prepared.placement_state)
        if not isinstance(prepared.placement_state, _MixedUnshardState):
            raise AssertionError(
                "Expected _MixedUnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        state = prepared.placement_state
        # A refill returns no full params (see PlacementUnshardResult).
        refill = prepared.persistent_buffers is not None
        gathered_by_rank = prepared.buffers[1].view(
            state.world_size,
            state.row_numel,
        )
        full_params: list[torch.Tensor | None] = [None] * sum(
            len(group.indices) for group in state.groups
        )
        buffers: list[torch.Tensor] = []
        persistent_buffers: list[torch.Tensor] = []
        buffer_offset = 0
        with _record_copy_out_if_eager():
            for group in state.groups:
                placement = group.prepared.placement
                num_buffers = _num_persistent_buffers(placement, len(group.indices))
                group_persistent_buffers = (
                    prepared.persistent_buffers[
                        buffer_offset : buffer_offset + num_buffers
                    ]
                    if prepared.persistent_buffers is not None
                    else None
                )
                buffer_offset += num_buffers
                group_gathered_by_rank = gathered_by_rank[
                    :, group.offset : group.offset + group.numel
                ]
                if isinstance(placement, BlockwiseQuantBucketedBlockShard):
                    gathered = group_gathered_by_rank
                    group.prepared.persistent = prepared.persistent
                    group.prepared.persistent_buffers = group_persistent_buffers
                    result = placement._finish_unshard_from_rank_rows(
                        group.prepared,
                        gathered,
                    )
                else:
                    if state.requires_dtype_reinterpretation:
                        gathered = group.prepared.buffers[1]
                        gathered.view(torch.uint8).view_as(
                            group_gathered_by_rank
                        ).copy_(group_gathered_by_rank)
                    else:
                        gathered = group_gathered_by_rank.contiguous().view(-1)
                    group_prepared = PlacementPreparedUnshard(
                        placement=placement,
                        buffers=[
                            group.prepared.buffers[0],
                            gathered,
                            *group.prepared.buffers[2:],
                        ],
                        placement_state=group.prepared.placement_state,
                        persistent=prepared.persistent,
                        persistent_buffers=group_persistent_buffers,
                    )
                    result = placement.finish_prepared_unshard(group_prepared)
                buffers.append(gathered)
                if not refill:
                    for index, full_param in zip(
                        group.indices,
                        result.full_params,
                        strict=True,
                    ):
                        full_params[index] = full_param
                buffers.extend(result.buffers)
                persistent_buffers.extend(result.persistent_buffers)

        if refill:
            return PlacementUnshardResult(
                full_params=[],
                buffers=buffers,
                persistent_buffers=persistent_buffers,
            )
        ordered_full_params: list[torch.Tensor] = []
        for full_param in full_params:
            if full_param is None:
                raise AssertionError("Mixed unshard did not produce every param.")
            ordered_full_params.append(full_param)
        return PlacementUnshardResult(
            full_params=ordered_full_params,
            buffers=buffers,
            persistent_buffers=persistent_buffers,
        )

    def _prepare_owner_rows_unshard(
        self,
        groups: list[_PlacementGroup],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard | None:
        """Unshard a bucket whose groups each live whole on one owner, as
        BucketedOwned's whole parameters do, in one dtype: every param lies
        contiguously in its owner's row of the gathered buffer, so the full
        params view those rows, as BucketedBlockShard's view its bucket, and
        refills gather straight into them. Each rank's row is its local storage
        (see _owner_rows_storage_layout), which it sends as is. Rows keep their
        own lengths, back to back (see _gathered_row_numels). None for any
        other bucket."""
        rank = mesh.get_local_rank()
        world_size = mesh.size()
        dtype = _unsharded_dtype(groups[0].infos[0])
        plan = _owner_rows_plan(groups, world_size)
        if plan is None or any(
            _unsharded_dtype(info) != dtype for group in groups for info in group.infos
        ):
            return None
        for group in groups:
            group.placement._validate_bucket_inputs(group.tensors, group.infos)
        layouts, offsets, owner_ranks, row_numels = plan
        # This rank's params, each with its offset in this rank's row.
        owned = [
            (tensor, offset + layout.param_offsets[index])
            for group, layout, offset, owner_rank in zip(
                groups, layouts, offsets, owner_ranks, strict=True
            )
            if owner_rank == rank
            for index, tensor in enumerate(group.tensors)
        ]
        device = groups[0].tensors[0].device
        transport = owner_rows_transport_plan(mesh, row_numels, device)
        gather = None
        if transport is not None:
            # The owner writes straight from its params' storage, casting on
            # the way, so there is no send row.
            row_start = _row_starts(row_numels)[rank]
            gather = _OwnerRowsGather(
                plan=transport[0],
                hosts=transport[1],
                buffers=owner_rows_transport_buffers(mesh, transport[1]),
                rank=rank,
                sources=[(tensor, row_start + start) for tensor, start in owned],
            )
            send = torch.empty(0, dtype=dtype, device=device)
        else:
            send = _owner_row_view(owned, row_numels[rank], dtype)
        if send is None:
            # The params are not in such storage (e.g. they are in another
            # dtype, or this rank owns none): copy them in.
            with _record_copy_in_if_eager():
                send = torch.zeros(
                    row_numels[rank], dtype=dtype, device=groups[0].tensors[0].device
                )
                foreach_copy_(
                    [send[start : start + tensor.numel()] for tensor, start in owned],
                    [tensor.reshape(-1) for tensor, _ in owned],
                )
        group_states = [
            _MixedUnshardGroupState(
                prepared=PlacementPreparedUnshard(
                    placement=group.placement,
                    buffers=[],
                    placement_state=BlockShard._UnshardState(
                        infos=group.infos,
                        layout=layout,
                        pg=mesh.get_group(),
                        debug_fqn=debug_fqn,
                    ),
                ),
                indices=tuple(group.indices),
                offset=offset,
                numel=layout.padded_segment_numel,
            )
            for group, layout, offset in zip(groups, layouts, offsets, strict=True)
        ]
        return PlacementPreparedUnshard(
            placement=self,
            buffers=[send],
            placement_state=_OwnerRowsUnshardState(
                groups=group_states,
                world_size=world_size,
                row_numels=row_numels,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                transport=gather,
            ),
        )

    def _run_owner_rows_unshard(
        self,
        prepared: PlacementPreparedUnshard,
        state: _OwnerRowsUnshardState,
        *,
        async_op: bool = False,
    ) -> dist.Work | None:
        if prepared.persistent_buffers is not None:
            # A refill gathers straight into the persistent rows.
            gathered = prepared.persistent_buffers[0]
        else:
            gathered = prepared.buffers[0].new_empty(
                sum(state.row_numels)
                if state.transport is None
                else state.transport.plan.gathered_numel
            )
            prepared.buffers.insert(1, gathered)
        if state.transport is not None:
            gather = state.transport
            with _record_comm_if_eager("FlexShard::owner_rows_gather", state.debug_fqn):
                return gather_owner_rows(
                    gather.plan,
                    gather.hosts,
                    gather.buffers,
                    rank=gather.rank,
                    row_start=_row_starts(state.row_numels)[gather.rank],
                    row_numel=state.row_numels[gather.rank],
                    sources=gather.sources,
                    gathered=gathered,
                    pg=state.pg,
                    async_op=async_op,
                )
        with _record_comm_if_eager("FlexShard::mixed_all_gather", state.debug_fqn):
            if len(set(state.row_numels)) == 1:
                return dist.all_gather_into_tensor(
                    output_tensor=gathered,
                    input_tensor=prepared.buffers[0],
                    group=state.pg,
                    async_op=async_op,
                )
            # Uneven rows: one broadcast per owner in one NCCL group, each
            # straight into its row. NCCL 2.30 fuses the group into an
            # all-gather-v that runs at the pace of the longest row;
            # NCCL_ALLGATHERV_ENABLE=0 keeps the broadcasts, much faster
            # across hosts.
            rows = list(gathered.split(state.row_numels))
            send = prepared.buffers[0]
            if not state.row_numels[0] and not state.row_numels[1]:
                # ProcessGroupNCCL2's allgather takes outputs whose first two
                # share an address for one aliased buffer and gathers equal
                # rows instead, and every empty tensor has address 0. So rank 0
                # sends one spare element, into scratch.
                rows[0] = gathered.new_empty(1)
                prepared.buffers.append(rows[0])
                if state.pg.rank() == 0:
                    send = rows[0]
            return dist.all_gather(rows, send, group=state.pg, async_op=async_op)

    def _finish_owner_rows_unshard(
        self,
        prepared: PlacementPreparedUnshard,
        state: _OwnerRowsUnshardState,
    ) -> PlacementUnshardResult:
        if prepared.persistent_buffers is not None:
            # Run refilled the persistent rows.
            return PlacementUnshardResult(
                full_params=[],
                persistent_buffers=list(prepared.persistent_buffers),
            )
        gathered = prepared.buffers[1]
        if prepared.persistent:
            # The first unshard, as for BucketedBlockShard: the gathered rows
            # come from the unshard stream, and persistent storage lives on the
            # current stream, so copy into it. Refills gather into it.
            rows = torch.empty_like(gathered)
            with _record_copy_out_if_eager():
                rows.copy_(gathered)
        else:
            rows = gathered
        row_starts = _row_starts(state.row_numels)
        full_params: list[torch.Tensor | None] = [None] * sum(
            len(group.indices) for group in state.groups
        )
        for group in state.groups:
            owner_rank = _one_hot_block_owner(
                group.prepared.placement, state.world_size
            )
            group_state = group.prepared.placement_state
            for index, (param_index, info) in enumerate(
                zip(group.indices, group_state.infos, strict=True)
            ):
                start = (
                    row_starts[owner_rank]
                    + group.offset
                    + group_state.layout.param_offsets[index]
                )
                full_params[param_index] = rows[start : start + info.global_numel].view(
                    info.global_shape
                )
        return PlacementUnshardResult(
            full_params=full_params,
            persistent_buffers=[rows] if prepared.persistent else [],
        )

    def _prepare_owner_rows_reduce_grad(
        self,
        groups: list[_PlacementGroup],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad | None:
        """Reduce the grads' gradient bucket as is, if they view one that
        mirrors the owner rows the params view (see
        ``BucketSpec.pre_backward_hook`` and _prepare_owner_rows_unshard):
        each grad already sits in its owner's row. None otherwise."""
        bucket = groups[0].tensors[0]._base
        plan = _owner_rows_plan(groups, mesh.size())
        if bucket is None or plan is None:
            return None
        layouts, offsets, owner_ranks, row_numels = plan
        transport = _owner_rows_reduce_transport(
            groups, owner_ranks, mesh, row_numels, bucket.device
        )
        numel = sum(row_numels) if transport is None else transport.plan.gathered_numel
        if (
            bucket.dtype != _grad_reduce_dtype(groups[0].infos[0])
            or bucket.shape != (numel,)
            or not bucket.is_contiguous()
        ):
            return None
        row_starts = _row_starts(row_numels)
        for group, layout, offset, owner_rank in zip(
            groups, layouts, offsets, owner_ranks, strict=True
        ):
            for index, (tensor, info) in enumerate(
                zip(group.tensors, group.infos, strict=True)
            ):
                if (
                    tensor._base is not bucket
                    or tensor.shape != info.global_shape
                    or not tensor.is_contiguous()
                    or tensor.storage_offset() - bucket.storage_offset()
                    != row_starts[owner_rank] + offset + layout.param_offsets[index]
                ):
                    return None
        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=[bucket],
            placement_state=_owner_rows_reduce_state(
                groups, layouts, offsets, row_numels, mesh, debug_fqn, transport
            ),
        )

    def _pack_owner_rows_reduce_grad(
        self,
        groups: list[_PlacementGroup],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad | None:
        """Copy the grads of a bucket whose groups each live whole on one
        owner into the owners' rows, laid out as the unshard's (see
        _prepare_owner_rows_unshard), when they don't already view such a
        gradient bucket. None for any other bucket."""
        plan = _owner_rows_plan(groups, mesh.size())
        if plan is None:
            return None
        layouts, offsets, owner_ranks, row_numels = plan
        dtype = _grad_reduce_dtype(groups[0].infos[0])
        if any(
            _grad_reduce_dtype(info) != dtype
            for group in groups
            for info in group.infos
        ):
            return None
        row_starts = _row_starts(row_numels)
        device = groups[0].tensors[0].device
        transport = _owner_rows_reduce_transport(
            groups, owner_ranks, mesh, row_numels, device
        )
        with _record_function_if_eager(
            "FlexShard::mixed_reduce_scatter_copy_in",
            debug_fqn,
        ):
            # The grads fill their rows but for the padding torch.compile adds
            # (see _gathered_row_numels), and the transport's tail padding up
            # to whole chunks, which no row holds.
            alloc = torch.zeros if torch.compiler.is_compiling() else torch.empty
            send = alloc(
                sum(row_numels) if transport is None else transport.plan.gathered_numel,
                dtype=dtype,
                device=device,
            )
            dsts: list[torch.Tensor] = []
            srcs: list[torch.Tensor] = []
            for group, layout, offset, owner_rank in zip(
                groups, layouts, offsets, owner_ranks, strict=True
            ):
                group.placement._validate_bucket_inputs(group.tensors, group.infos)
                for index, (tensor, info) in enumerate(
                    zip(group.tensors, group.infos, strict=True)
                ):
                    if tensor.shape != info.global_shape:
                        raise ValueError(
                            f"FlexShard full gradient {info.fqn!r} has shape "
                            f"{tuple(tensor.shape)}, expected "
                            f"{tuple(info.global_shape)}."
                        )
                    start = (
                        row_starts[owner_rank] + offset + layout.param_offsets[index]
                    )
                    dsts.append(send[start : start + tensor.numel()])
                    srcs.append(tensor.reshape(-1))
            foreach_copy_(dsts, srcs)
        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=[send],
            placement_state=_owner_rows_reduce_state(
                groups, layouts, offsets, row_numels, mesh, debug_fqn, transport
            ),
        )

    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad:
        world_size = mesh.size()
        groups = _group_tensors_by_placement(tensors, infos, world_size)
        _validate_mixed_groups(groups)
        owner_rows = self._prepare_owner_rows_reduce_grad(groups, mesh, debug_fqn)
        if owner_rows is None:
            owner_rows = self._pack_owner_rows_reduce_grad(groups, mesh, debug_fqn)
        if owner_rows is not None:
            return owner_rows
        prepared_groups: list[tuple[PlacementPreparedReduceGrad, list[int], int]] = []
        group_buffers: list[torch.Tensor] = []
        dtype: torch.dtype | None = None
        device: torch.device | None = None

        with _record_function_if_eager(
            "FlexShard::mixed_reduce_scatter_copy_in",
            debug_fqn,
        ):
            for group in groups:
                if _one_hot_block_owner(group.placement, world_size) is not None:
                    # A group with one owner fills only the owner's row: it is
                    # packed straight into the bucket's send buffer below,
                    # rather than into a world-size buffer of its own, which
                    # would be zeros but for that row.
                    group.placement._validate_bucket_inputs(group.tensors, group.infos)
                    layout = group.placement._padded_bucket_layout(group.infos)
                    prepared = PlacementPreparedReduceGrad(
                        placement=group.placement,
                        buffers=[],
                        placement_state=BlockShard._ReduceGradState(
                            infos=group.infos,
                            layout=layout,
                            pg=mesh.get_group(),
                            debug_fqn=debug_fqn,
                        ),
                    )
                    group_numel = layout.padded_segment_numel
                    group_dtype = _grad_reduce_dtype(group.infos[0])
                    group_device = group.tensors[0].device
                elif isinstance(
                    group.placement, BlockwiseQuantBucketedBlockShard
                ) and group.placement._packs_reduce_scatter_grad_in_one_pass(
                    group.tensors, _grad_reduce_dtype(group.infos[0])
                ):
                    # The grads are packed straight into the bucket's send
                    # buffer below, in one pass, rather than into a
                    # world-size buffer of their own first.
                    group_numel = max(
                        group.placement._bucket_layout(group.infos[0]).rank_numels
                    )
                    prepared = PlacementPreparedReduceGrad(
                        placement=group.placement,
                        buffers=[],
                        placement_state=BucketedBlockShard._ReduceGradState(
                            infos=group.infos,
                            rank=mesh.get_local_rank(),
                            pg=mesh.get_group(),
                            debug_fqn=debug_fqn,
                            padded_segment_numel=group_numel,
                        ),
                    )
                    group_dtype = _grad_reduce_dtype(group.infos[0])
                    group_device = group.tensors[0].device
                else:
                    prepared = group.placement.prepare_reduce_grad(
                        group.tensors,
                        group.infos,
                        mesh,
                        debug_fqn,
                    )
                    send = prepared.buffers[0]
                    if send.numel() % world_size != 0:
                        raise AssertionError(
                            "Mixed reduce-scatter subgroup send size must be "
                            f"divisible by world size {world_size}, got "
                            f"{send.numel()}."
                        )
                    group_numel = send.numel() // world_size
                    group_dtype = send.dtype
                    group_device = send.device
                if dtype is None:
                    dtype = group_dtype
                    device = group_device
                elif group_dtype != dtype or group_device != device:
                    raise ValueError(
                        "Mixed FlexShard reduce-grad requires one send dtype "
                        f"and device per bucket, but got {dtype}/{device} and "
                        f"{group_dtype}/{group_device}."
                    )
                prepared_groups.append((prepared, group.indices, group_numel))
                group_buffers.extend(prepared.buffers)

            if dtype is None or device is None:
                raise AssertionError("Expected at least one mixed bucket group.")
            (
                offsets,
                owner_ranks,
                row_numel,
                max_exclusive_payload_numel,
            ) = _plan_group_offsets(
                [prepared.placement for prepared, _, _ in prepared_groups],
                [numel for _, _, numel in prepared_groups],
                world_size,
            )
            group_states = [
                _MixedReduceGradGroupState(
                    prepared=prepared,
                    indices=indices,
                    offset=offset,
                    numel=numel,
                )
                for (prepared, indices, numel), offset in zip(
                    prepared_groups,
                    offsets,
                    strict=True,
                )
            ]
            send = torch.empty(
                world_size * row_numel,
                dtype=dtype,
                device=device,
            )
            send_rows = send.view(world_size, row_numel)
            if max_exclusive_payload_numel:
                send_rows[:, :max_exclusive_payload_numel].zero_()
            for group, group_state, owner_rank in zip(
                groups,
                group_states,
                owner_ranks,
                strict=True,
            ):
                start = group_state.offset
                end = start + group_state.numel
                if not group_state.prepared.buffers:
                    group.placement._pack_reduce_scatter_grad(
                        group.tensors,
                        group.infos,
                        world_size,
                        send_rows[:, start:end],
                    )
                    continue
                group_send = group_state.prepared.buffers[0].view(
                    world_size,
                    group_state.numel,
                )
                if owner_rank is None:
                    send_rows[:, start:end].copy_(group_send)
                else:
                    send_rows[owner_rank, start:end].copy_(group_send[owner_rank])

        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=[send, *group_buffers],
            placement_state=_MixedReduceGradState(
                groups=group_states,
                world_size=world_size,
                row_numel=row_numel,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
            ),
        )

    def reduces_grads_in_param_layout(
        self, infos: list[ParamInfo], world_size: int
    ) -> bool:
        """Owner rows do (see _prepare_owner_rows_reduce_grad) when the
        unsharded params are in the dtype their grads accumulate and reduce
        in, so a gradient buffer in the dtype of the storage they view keeps
        autograd's accumulation."""
        dtype = _unsharded_dtype(infos[0])
        return all(
            _one_hot_block_owner(
                _collective_placement(
                    info.placement, info.fqn, info.global_shape, world_size
                ),
                world_size,
            )
            is not None
            and _unsharded_dtype(info) == dtype
            and info.unsharded_grad_dtype in (None, dtype)
            and _grad_reduce_dtype(info) == dtype
            for info in infos
        )

    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
        reduction: GradientReduction,
    ) -> PlacementReduceGradResult:
        if not isinstance(prepared.placement_state, _MixedReduceGradState):
            raise AssertionError(
                "Expected _MixedReduceGradState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        state = prepared.placement_state
        send = prepared.buffers[0]
        recv = torch.empty(
            state.row_numel,
            dtype=send.dtype if state.transport is None else state.transport.recv_dtype,
            device=send.device,
        )
        uneven_rows = state.row_numels is not None and len(set(state.row_numels)) > 1
        with _record_comm_if_eager(
            "FlexShard::mixed_reduce_scatter",
            state.debug_fqn,
        ):
            if state.transport is not None:
                reduce = state.transport
                reduce_owner_rows(
                    reduce.plan,
                    reduce.hosts,
                    reduce.buffers,
                    rank=reduce.rank,
                    row_start=_row_starts(state.row_numels)[reduce.rank],
                    send=send,
                    recv=recv,
                    reduction=reduction,
                    pg=state.pg,
                )
            elif uneven_rows:
                # One reduce per owner in one NCCL group, each straight from
                # its row.
                reduce_scatter_grads_uneven(
                    recv, list(send.split(state.row_numels)), reduction, state.pg
                )
            else:
                reduce_scatter_grads(recv, send, reduction, state.pg)
            for group in state.groups:
                _release_group_scratch_lease(group.prepared)

        sharded_grads: list[torch.Tensor | None] = [None] * sum(
            len(group.indices) for group in state.groups
        )
        with _record_function_if_eager(
            "FlexShard::mixed_reduce_scatter_copy_out",
            state.debug_fqn,
        ):
            for group in state.groups:
                if (
                    uneven_rows
                    and _one_hot_block_owner(group.prepared.placement, state.world_size)
                    != state.pg.rank()
                ):
                    # Another rank owns the group, and this rank's row, which
                    # may be shorter, holds none of it.
                    group_recv = recv[:0]
                else:
                    group_recv = recv.narrow(0, group.offset, group.numel)
                group_grads = _finish_mixed_reduce_group(
                    group.prepared,
                    group_recv,
                )
                for index, sharded_grad in zip(
                    group.indices,
                    group_grads,
                    strict=True,
                ):
                    sharded_grads[index] = sharded_grad

        ordered_sharded_grads: list[torch.Tensor] = []
        for sharded_grad in sharded_grads:
            if sharded_grad is None:
                raise AssertionError("Mixed reduce-grad did not produce every grad.")
            ordered_sharded_grads.append(sharded_grad)
        return PlacementReduceGradResult(ordered_sharded_grads, [recv])


def _prepare_mixed_unshard_group(
    placement: Placement,
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    debug_fqn: str | None,
) -> PlacementPreparedUnshard:
    if isinstance(placement, BlockwiseQuantBucketedBlockShard):
        return placement._prepare_local_unshard_payload(
            tensors,
            infos,
            mesh,
            debug_fqn,
        )
    if isinstance(placement, Shard):
        # The mixed all-gather fills the group's gather buffer, even on a
        # 1-rank mesh where Shard alone skips its all-gather.
        return placement._prepare_unshard_bucket(
            tensors, infos, mesh, debug_fqn, skip_single_rank_all_gather=False
        )
    return placement.prepare_unshard_bucket(
        tensors,
        infos,
        mesh,
        debug_fqn,
    )


def _zero_padding(
    tensor: torch.Tensor,
    ranges: tuple[tuple[int, int], ...],
) -> None:
    slices = [tensor.narrow(0, offset, numel) for offset, numel in ranges]
    if torch.compiler.is_compiling():
        for slice_ in slices:
            slice_.zero_()
    elif slices:
        torch._foreach_zero_(slices)


def _build_mixed_unshard_layout(
    prepared_groups: list[tuple[PlacementPreparedUnshard, tuple[int, ...]]],
    world_size: int,
) -> _MixedUnshardLayout:
    dtypes = {group_prepared.buffers[0].dtype for group_prepared, _ in prepared_groups}
    requires_dtype_reinterpretation = len(dtypes) > 1
    dtype = torch.uint8 if requires_dtype_reinterpretation else next(iter(dtypes))
    group_numels = [
        (
            group_prepared.buffers[0].numel() * group_prepared.buffers[0].element_size()
            if requires_dtype_reinterpretation
            else group_prepared.buffers[0].numel()
        )
        for group_prepared, _ in prepared_groups
    ]
    group_placements = [prepared.placement for prepared, _ in prepared_groups]
    quant_transport_groups = [
        isinstance(prepared.placement, BlockwiseQuantBucketedBlockShard)
        and prepared.buffers[0].dtype == torch.uint8
        for prepared, _ in prepared_groups
    ]
    align_quant_transport = any(quant_transport_groups)
    group_alignments = [
        _VEC_ALIGN_BYTES if is_quant_transport else 1
        for is_quant_transport in quant_transport_groups
    ]
    row_alignment = _VEC_ALIGN_BYTES if align_quant_transport else 1
    offsets, owner_ranks, row_numel, max_exclusive_payload_numel = _plan_group_offsets(
        group_placements,
        group_numels,
        world_size,
        group_alignments=group_alignments,
        row_alignment=row_alignment,
    )
    groups = tuple(
        _MixedUnshardGroupLayout(
            offset=offset,
            numel=numel,
            owner_rank=owner_rank,
        )
        for offset, numel, owner_rank in zip(
            offsets,
            group_numels,
            owner_ranks,
            strict=True,
        )
    )
    padding_ranges: list[tuple[int, int]] = []
    if max_exclusive_payload_numel:
        padding_ranges.append((0, max_exclusive_payload_numel))
    next_padding_start = max_exclusive_payload_numel
    for group in groups:
        if group.owner_rank is not None:
            continue
        if next_padding_start < group.offset:
            padding_ranges.append(
                (next_padding_start, group.offset - next_padding_start)
            )
        next_padding_start = group.offset + group.numel
    if next_padding_start < row_numel:
        padding_ranges.append((next_padding_start, row_numel - next_padding_start))
    return _MixedUnshardLayout(
        groups=groups,
        dtype=dtype,
        requires_dtype_reinterpretation=requires_dtype_reinterpretation,
        row_numel=row_numel,
        padding_ranges=tuple(padding_ranges),
    )


def _group_tensors_by_placement(
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    world_size: int,
    key_fn: Callable[[ParamInfo], object] | None = None,
) -> list[_PlacementGroup]:
    groups: list[_PlacementGroup] = []
    for index, (tensor, info) in enumerate(zip(tensors, infos, strict=True)):
        placement = _collective_placement(
            info.placement,
            info.fqn,
            info.global_shape,
            world_size,
        )
        # Quantized pack plans span the whole member and index infos by
        # position, and quantization reads each param in its own dtype, so
        # never split a quantized member.
        key = (
            None
            if key_fn is None or isinstance(placement, BlockwiseQuantBucketedBlockShard)
            else key_fn(info)
        )
        for group in groups:
            if placement == group.placement and key == group.key:
                group.indices.append(index)
                group.tensors.append(tensor)
                group.infos.append(info)
                break
        else:
            groups.append(
                _PlacementGroup(
                    placement=placement,
                    key=key,
                    indices=[index],
                    tensors=[tensor],
                    infos=[info],
                )
            )
    return groups


def _group_local_params_by_placement(
    local_params: list[tuple[str, torch.nn.Parameter, Placement]],
) -> list[list[tuple[str, torch.nn.Parameter, Placement]]]:
    groups: list[list[tuple[str, torch.nn.Parameter, Placement]]] = []
    for local_param in local_params:
        placement = local_param[2]
        for group in groups:
            if placement == group[0][2]:
                group.append(local_param)
                break
        else:
            groups.append([local_param])
    return groups


def _num_persistent_buffers(placement: Placement, num_params: int) -> int:
    """persistent buffers a group's persistent unshard returns (see finish)."""
    # Blockwise quantized groups back all params with one compact buffer;
    # Shard and BlockShard groups back each param with its own buffer.
    return 1 if isinstance(placement, BlockwiseQuantBucketedBlockShard) else num_params


def _collective_placement(
    placement: Placement,
    fqn: str,
    global_shape: torch.Size,
    world_size: int,
) -> Placement:
    if isinstance(placement, _MixedBucketedOwned):
        return placement._lower_to_block_shards(
            {fqn: tuple(global_shape)},
            world_size,
        )[fqn]
    if isinstance(placement, _MixedBucketMember):
        return placement._collective_placement
    return placement


def _one_hot_block_owner(
    placement: Placement,
    world_size: int,
) -> int | None:
    if not isinstance(placement, BlockShard):
        return None
    counts = placement.blocks_per_rank
    if (
        len(counts) != world_size
        or counts.count(1) != 1
        or any(count not in (0, 1) for count in counts)
    ):
        return None
    return counts.index(1)


def _owner_row_view(
    owned: list[tuple[torch.Tensor, int]],
    row_numel: int,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """This rank's row of the gathered buffer as a view of the storage its
    params live in, if they sit there at their offsets in the row, in
    ``dtype``, and the storage holds the whole row, as
    MixedBucketPlacement._owner_rows_storage_layout lays them out. Eager only."""
    if (
        torch.compiler.is_compiling()
        or not owned
        or any(
            tensor.dtype != dtype or not tensor.is_contiguous() for tensor, _ in owned
        )
    ):
        return None
    first, first_start = owned[0]
    storage = first.untyped_storage()
    row_start = first.storage_offset() - first_start
    if row_start < 0 or (row_start + row_numel) * dtype.itemsize > storage.nbytes():
        return None
    if any(
        tensor.untyped_storage().data_ptr() != storage.data_ptr()
        or tensor.storage_offset() != row_start + start
        for tensor, start in owned
    ):
        return None
    return first.new_empty(0).set_(storage, row_start, (row_numel,), (1,))


def _plan_group_offsets(
    placements: list[Placement],
    group_numels: list[int],
    world_size: int,
    *,
    group_alignments: list[int] | None = None,
    row_alignment: int = 1,
) -> tuple[list[int], list[int | None], int, int]:
    if group_alignments is None:
        group_alignments = [1] * len(placements)
    if len(group_alignments) != len(placements):
        raise AssertionError(
            "Expected one mixed-bucket alignment per placement, got "
            f"{len(group_alignments)} for {len(placements)} placements."
        )
    owner_numels = [0] * world_size
    offsets = []
    owner_ranks = []
    regular_numel = 0
    regular_alignment = 1
    for placement, group_numel, alignment in zip(
        placements,
        group_numels,
        group_alignments,
        strict=True,
    ):
        owner_rank = _one_hot_block_owner(placement, world_size)
        owner_ranks.append(owner_rank)
        if owner_rank is None:
            regular_numel = _align_up(regular_numel, alignment)
            offsets.append(regular_numel)
            regular_numel += group_numel
            regular_alignment = max(regular_alignment, alignment)
        else:
            owner_numels[owner_rank] = _align_up(
                owner_numels[owner_rank],
                alignment,
            )
            offsets.append(owner_numels[owner_rank])
            owner_numels[owner_rank] += group_numel

    regular_base = _align_up(
        max(owner_numels, default=0),
        regular_alignment,
    )
    offsets = [
        offset + regular_base if owner_rank is None else offset
        for offset, owner_rank in zip(offsets, owner_ranks, strict=True)
    ]
    return (
        offsets,
        owner_ranks,
        _align_up(regular_base + regular_numel, row_alignment),
        regular_base,
    )


def _owner_row_numels(
    owner_ranks: list[int | None],
    offsets: list[int],
    group_numels: list[int],
    world_size: int,
) -> list[int]:
    """Each rank's row length in a bucket whose groups each live whole on one
    owner: where its last group ends, 0 for a rank owning none."""
    row_numels = [0] * world_size
    for owner_rank, offset, numel in zip(
        owner_ranks, offsets, group_numels, strict=True
    ):
        if owner_rank is None:
            raise AssertionError("Owner rows require one owner per group.")
        row_numels[owner_rank] = max(row_numels[owner_rank], offset + numel)
    return row_numels


def _gathered_row_numels(row_numels: list[int]) -> tuple[int, ...]:
    """The gathered buffer's rows, back to back: each at its owner's length, so
    a bucket holds no padding and uneven rows take one broadcast and one reduce
    per owner. Under torch.compile every row takes the longest's length, so
    one all-gather and one reduce-scatter trace as before."""
    if torch.compiler.is_compiling():
        return (max(row_numels),) * len(row_numels)
    return tuple(row_numels)


def _row_starts(row_numels: tuple[int, ...]) -> list[int]:
    return list(itertools.accumulate(row_numels[:-1], initial=0))


def _owner_rows_plan(
    groups: list[_PlacementGroup],
    world_size: int,
) -> tuple[list[Any], list[int], list[int], tuple[int, ...]] | None:
    """The owner-rows layout of a bucket whose groups each live whole on one
    owner: each group's padded layout, its offset in its owner's row, the
    owners, and the gathered rows' lengths. None for any other bucket."""
    if any(
        _one_hot_block_owner(group.placement, world_size) is None for group in groups
    ):
        return None
    layouts = [group.placement._padded_bucket_layout(group.infos) for group in groups]
    group_numels = [layout.padded_segment_numel for layout in layouts]
    offsets, owner_ranks, _, _ = _plan_group_offsets(
        [group.placement for group in groups], group_numels, world_size
    )
    row_numels = _gathered_row_numels(
        _owner_row_numels(owner_ranks, offsets, group_numels, world_size)
    )
    return layouts, offsets, owner_ranks, row_numels


def _owner_rows_reduce_state(
    groups: list[_PlacementGroup],
    layouts: list[Any],
    offsets: list[int],
    row_numels: tuple[int, ...],
    mesh: DeviceMesh,
    debug_fqn: str | None,
    transport: _OwnerRowsReduce | None = None,
) -> _MixedReduceGradState:
    return _MixedReduceGradState(
        groups=[
            _MixedReduceGradGroupState(
                prepared=PlacementPreparedReduceGrad(
                    placement=group.placement,
                    buffers=[],
                    placement_state=BlockShard._ReduceGradState(
                        infos=group.infos,
                        layout=layout,
                        pg=mesh.get_group(),
                        debug_fqn=debug_fqn,
                    ),
                ),
                indices=group.indices,
                offset=offset,
                numel=layout.padded_segment_numel,
            )
            for group, layout, offset in zip(groups, layouts, offsets, strict=True)
        ],
        world_size=mesh.size(),
        row_numel=row_numels[mesh.get_local_rank()],
        pg=mesh.get_group(),
        debug_fqn=debug_fqn,
        row_numels=row_numels,
        transport=transport,
    )


def _owner_rows_reduce_transport(
    groups: list[_PlacementGroup],
    owner_ranks: list[int],
    mesh: DeviceMesh,
    row_numels: tuple[int, ...],
    device: torch.device,
) -> _OwnerRowsReduce | None:
    """The reduce through equal chunks for owner rows, if the transport takes
    them (see owner_rows_transport_plan). This rank's row comes back in its
    sharded grads' dtype (the params' dtype, or their explicit grad_dtype),
    so the cast happens on the way; in the reduce dtype if they disagree."""
    transport = owner_rows_transport_plan(mesh, row_numels, device)
    if transport is None:
        return None
    rank = mesh.get_local_rank()
    dtypes = {
        info.grad_dtype_override if info.has_explicit_grad_dtype else info.dtype
        for group, owner_rank in zip(groups, owner_ranks, strict=True)
        if owner_rank == rank
        for info in group.infos
    }
    recv_dtype = (
        next(iter(dtypes))
        if len(dtypes) == 1 and None not in dtypes
        else _grad_reduce_dtype(groups[0].infos[0])
    )
    return _OwnerRowsReduce(
        plan=transport[0],
        hosts=transport[1],
        buffers=owner_rows_transport_buffers(mesh, transport[1]),
        rank=rank,
        recv_dtype=recv_dtype,
    )


def _validate_mixed_groups(groups: list[_PlacementGroup]) -> None:
    if not groups:
        raise ValueError("Mixed FlexShard bucket collectives require parameters.")
    for group in groups:
        placement = group.placement
        if not (
            isinstance(placement, BlockShard)
            or isinstance(placement, Shard)
            and placement.dim == 0
            or isinstance(placement, BlockwiseQuantBucketedBlockShard)
        ):
            raise ValueError(
                "Mixed FlexShard bucket collectives currently support only "
                "Shard(0), BlockShard, and blockwise quantized subgroups."
            )


def _release_group_scratch_lease(prepared: PlacementPreparedReduceGrad) -> None:
    lease = getattr(prepared.placement_state, "scratch_lease", None)
    if lease is not None:
        lease.release()


def _finish_mixed_reduce_group(
    prepared: PlacementPreparedReduceGrad,
    recv: torch.Tensor,
) -> list[torch.Tensor]:
    placement = prepared.placement
    state = prepared.placement_state
    if isinstance(placement, Shard):
        return placement._unpack_reduce_scatter_grad(
            recv,
            state.infos,
            state.layout,
            state.rank,
            state.world_size,
        )
    if isinstance(placement, BlockShard):
        return placement._unpack_reduce_scatter_grad(recv, state)
    if isinstance(placement, BlockwiseQuantBucketedBlockShard):
        bucket_layout = placement._bucket_layout(state.infos[0])
        rank_offset = bucket_layout.rank_offsets[state.rank]
        return [
            recv[
                placement._param_layout(info).local_global_offset
                - rank_offset : placement._param_layout(info).local_global_offset
                - rank_offset
                + info.local_numel
            ].view(info.local_shape)
            for info in state.infos
        ]
    raise TypeError(
        "Mixed FlexShard reduce-grad supports only Shard(0), BlockShard, "
        "and blockwise quantized placements, "
        f"but got {type(placement).__name__}."
    )


def make_mixed_placement_fn(
    owned_rank_by_fqn: Mapping[str, int],
    blocks_per_rank_by_fqn: Mapping[str, tuple[int, ...]],
) -> PlacementFn:
    def placement_fn(
        named_params: list[tuple[str, nn.Parameter]],
        _mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        owned_params = [
            (fqn, param) for fqn, param in named_params if fqn in owned_rank_by_fqn
        ]
        segments_by_fqn = {
            fqn: [
                BucketedOwnedSegmentSpec(
                    name=f"{fqn}#full",
                    fqn=fqn,
                    param_offset=0,
                    numel=param.numel(),
                    owner_rank=owned_rank_by_fqn[fqn],
                    storage_order=order,
                )
            ]
            for order, (fqn, param) in enumerate(owned_params)
        }
        mixed = MixedBucketPlacement(segments_by_fqn)
        placements: dict[str, tuple[Placement, ...]] = {}
        for fqn, _param in named_params:
            if fqn in blocks_per_rank_by_fqn:
                placement = mixed.block_shard(
                    blocks_per_rank=blocks_per_rank_by_fqn[fqn],
                    dim=0,
                )
            elif fqn in owned_rank_by_fqn:
                placement = mixed.bucketed_owned
            else:
                placement = mixed.shard0
            placements[fqn] = (placement,)
        return placements

    return placement_fn
