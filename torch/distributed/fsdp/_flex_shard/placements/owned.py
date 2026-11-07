# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace
from typing import Literal, TYPE_CHECKING
from typing_extensions import override

from ..placement_contract import (
    BucketStorageLayout,
    GlobalLayout,
    Placement,
    PlacementPreparedReduceGrad,
    PlacementPreparedUnshard,
    PlacementReduceGradResult,
    PlacementUnshardResult,
)
from ..utils import assign_to_hosts_then_ranks, mesh_rank_hosts
from .block_shard import BlockShard


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.distributed.device_mesh import DeviceMesh

    from ..bucket_storage import ParamInfo, PlacementFn
    from ..placement_contract import GradientReduction


@dataclass(frozen=True)
class BucketedOwnedSegmentSpec:
    """Flat slice of one parameter owned by one rank inside a BucketedOwned bucket."""

    name: str
    fqn: str
    param_offset: int
    numel: int
    owner_rank: int
    storage_order: int = 0


BucketedOwnedViewKind = Literal["full_param", "expert_block"]


class BucketedOwned(Placement):
    """Bucket-global owner-partition placement.

    Supported whole-tensor and complete dim-0 slab layouts lower privately to
    per-parameter ``BlockShard`` placements.
    """

    def __init__(
        self,
        segments_by_fqn: dict[str, list[BucketedOwnedSegmentSpec]],
        *,
        view_kind: BucketedOwnedViewKind = "full_param",
    ) -> None:
        if view_kind not in ("full_param", "expert_block"):
            raise ValueError(
                "BucketedOwned view_kind must be 'full_param' or 'expert_block', "
                f"but got {view_kind!r}."
            )
        normalized: dict[str, tuple[BucketedOwnedSegmentSpec, ...]] = {}
        for fqn, segments in segments_by_fqn.items():
            if not segments:
                raise ValueError(
                    f"BucketedOwned requires at least one segment for {fqn!r}."
                )
            normalized_segments = tuple(
                sorted(segments, key=lambda segment: segment.param_offset)
            )
            for segment in normalized_segments:
                if segment.fqn != fqn:
                    raise ValueError(
                        f"BucketedOwned segment {segment.name!r} is under key "
                        f"{fqn!r} but names FQN {segment.fqn!r}."
                    )
                if segment.numel <= 0:
                    raise ValueError(
                        f"BucketedOwned segment {segment.name!r} has invalid "
                        f"numel {segment.numel}."
                    )
            normalized[fqn] = normalized_segments
        self.segments_by_fqn = normalized
        self.view_kind = view_kind
        self._hash_key = tuple(
            (fqn, segments) for fqn, segments in sorted(normalized.items())
        )
        self._standalone_mixed_member: Placement | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BucketedOwned):
            return False
        return self.view_kind == other.view_kind and self._hash_key == other._hash_key

    def __hash__(self) -> int:
        return hash((type(self), self.view_kind, self._hash_key))

    def __repr__(self) -> str:
        return (
            f"BucketedOwned(num_params={len(self.segments_by_fqn)}, "
            f"view_kind={self.view_kind!r})"
        )

    @staticmethod
    def _validate_shape(fqn: str, shape: Sequence[int]) -> tuple[int, ...]:
        normalized = tuple(shape)
        if not normalized:
            raise ValueError(
                f"BucketedOwned cannot lower scalar parameter {fqn!r} to BlockShard."
            )
        if any(size < 0 for size in normalized):
            raise ValueError(
                f"BucketedOwned shape for {fqn!r} must be non-negative, but got "
                f"{shape}."
            )
        return normalized

    @staticmethod
    def _validate_segment(
        fqn: str,
        segment: BucketedOwnedSegmentSpec,
        world_size: int,
    ) -> None:
        if segment.fqn != fqn:
            raise ValueError(
                f"BucketedOwned segment {segment.name!r} is under key {fqn!r} "
                f"but names FQN {segment.fqn!r}."
            )
        if segment.param_offset < 0:
            raise ValueError("BucketedOwned param_offset must be non-negative.")
        if segment.numel <= 0:
            raise ValueError("BucketedOwned numel must be positive.")
        if segment.owner_rank < 0 or segment.owner_rank >= world_size:
            raise ValueError(
                f"BucketedOwned owner for segment {segment.name!r} must be in "
                f"[0, {world_size}), but got {segment.owner_rank}."
            )

    @staticmethod
    def _one_hot_blocks(owner_rank: int, world_size: int) -> tuple[int, ...]:
        return tuple(1 if rank == owner_rank else 0 for rank in range(world_size))

    @staticmethod
    def _blocks_per_rank_from_dim0_slices(
        fqn: str,
        shape: tuple[int, ...],
        segments: tuple[BucketedOwnedSegmentSpec, ...],
        world_size: int,
    ) -> tuple[int, ...]:
        slab_numel = math.prod(shape[1:])
        if slab_numel <= 0 or len(segments) != shape[0]:
            raise NotImplementedError(
                f"BucketedOwned {fqn!r} must contain one complete dim-0 slab "
                "per segment."
            )
        owners = []
        for slab_index, segment in enumerate(segments):
            if (
                segment.param_offset != slab_index * slab_numel
                or segment.numel != slab_numel
            ):
                raise NotImplementedError(
                    f"BucketedOwned {fqn!r} must contain equal complete dim-0 "
                    "slabs; partial segments cannot lower to BlockShard."
                )
            owners.append(segment.owner_rank)
        if owners != sorted(owners):
            raise NotImplementedError(
                f"BucketedOwned {fqn!r} requires contiguous, nondecreasing "
                "dim-0 slab owners to lower to BlockShard."
            )
        return tuple(owners.count(rank) for rank in range(world_size))

    def _lower_to_block_shards(
        self,
        logical_shapes_by_fqn: Mapping[str, Sequence[int]],
        world_size: int,
    ) -> dict[str, BlockShard]:
        """Lower supported ownership declarations to per-FQN BlockShard."""
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, but got {world_size}.")

        lowered: dict[str, BlockShard] = {}
        for fqn, shape_value in logical_shapes_by_fqn.items():
            if fqn not in self.segments_by_fqn:
                raise ValueError(f"BucketedOwned missing segments for {fqn!r}.")
            shape = self._validate_shape(fqn, shape_value)
            segments = self.segments_by_fqn[fqn]
            for segment in segments:
                self._validate_segment(fqn, segment, world_size)

            logical_numel = math.prod(shape)
            if (
                len(segments) == 1
                and segments[0].param_offset == 0
                and segments[0].numel == logical_numel
            ):
                blocks_per_rank = self._one_hot_blocks(
                    segments[0].owner_rank,
                    world_size,
                )
            else:
                blocks_per_rank = self._blocks_per_rank_from_dim0_slices(
                    fqn,
                    shape,
                    segments,
                    world_size,
                )
            lowered[fqn] = BlockShard(
                blocks_per_rank=blocks_per_rank,
                dim=0,
            )

        return lowered

    def _mixed_member(self) -> Placement:
        from .mixed_bucket import MixedBucketPlacement

        if self._standalone_mixed_member is None:
            mixed = MixedBucketPlacement(
                {fqn: list(segments) for fqn, segments in self.segments_by_fqn.items()}
            )
            self._standalone_mixed_member = mixed.bucketed_owned
        return self._standalone_mixed_member

    def _member_infos(self, infos: list[ParamInfo]) -> list[ParamInfo]:
        member = self._mixed_member()
        return [replace(info, placements=(member,)) for info in infos]

    @override
    def compute_local_shape(
        self,
        global_shape: torch.Size,
        rank: int,
        world_size: int,
    ) -> torch.Size:
        raise NotImplementedError(
            "BucketedOwned local shape is FQN-dependent; use bucket_storage_layout()."
        )

    @override
    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "BucketedOwned local shard is FQN-dependent; use copy_param_to_storage()."
        )

    @override
    def global_layout(
        self, info: ParamInfo, rank: int, world_size: int
    ) -> GlobalLayout:
        block_shard = self._lower_to_block_shards(
            {info.fqn: tuple(info.global_shape)},
            world_size,
        )[info.fqn]
        return block_shard.global_layout(info, rank, world_size)

    @override
    def bucket_storage_layout(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        mesh: DeviceMesh,
    ) -> BucketStorageLayout | None:
        expected = {fqn for fqn, _ in named_params}
        actual = set(self.segments_by_fqn)
        if expected != actual:
            raise ValueError(
                "BucketedOwned segment map must match the bucket FQNs exactly: "
                f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )
        for fqn, _ in named_params:
            placements = param_placements[fqn]
            if placements != (self,):
                raise ValueError(
                    "BucketedOwned requires the same placement instance for every "
                    f"parameter in a bucket; {fqn!r} uses {placements!r}."
                )
        member = self._mixed_member()
        return member.bucket_storage_layout(
            named_params,
            {fqn: (member,) for fqn, _ in named_params},
            mesh,
        )

    @override
    def copy_param_to_storage(
        self,
        byte_storage: torch.Tensor,
        info: ParamInfo,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> None:
        placement = self._lower_to_block_shards(
            {info.fqn: tuple(info.global_shape)},
            world_size,
        )[info.fqn]
        placement.copy_param_to_storage(
            byte_storage,
            info,
            param,
            rank,
            world_size,
        )

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        member = self._mixed_member()
        prepared = member.prepare_unshard_bucket(
            tensors,
            self._member_infos(infos),
            mesh,
            debug_fqn,
        )
        return PlacementPreparedUnshard(
            placement=self,
            buffers=prepared.buffers,
            placement_state=prepared.placement_state,
        )

    @override
    def run_prepared_unshard(
        self, prepared: PlacementPreparedUnshard, *, async_op: bool = False
    ) -> dist.Work | None:
        return self._mixed_member().run_prepared_unshard(prepared, async_op=async_op)

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        return self._mixed_member().finish_prepared_unshard(prepared)

    @override
    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad:
        member = self._mixed_member()
        prepared = member.prepare_reduce_grad(
            tensors,
            self._member_infos(infos),
            mesh,
            debug_fqn,
        )
        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=prepared.buffers,
            placement_state=prepared.placement_state,
        )

    @override
    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
        reduction: GradientReduction,
    ) -> PlacementReduceGradResult:
        return self._mixed_member().reduce_prepared_grad(prepared, reduction)

    @override
    def reduces_grads_in_param_layout(
        self, infos: list[ParamInfo], world_size: int
    ) -> bool:
        return self._mixed_member().reduces_grads_in_param_layout(
            self._member_infos(infos), world_size
        )


def _assign_params_to_ranks(
    named_params: list[tuple[str, nn.Parameter]],
    world_size: int,
    hosts: Sequence[int] | None = None,
    carried_loads: Sequence[int] | None = None,
) -> dict[str, int]:
    """Greedily assign each full parameter, largest first: with ``hosts``
    (each rank's host), to hosts first and then ranks within them (see
    ``assign_to_hosts_then_ranks``); without, to the least-loaded rank.
    ``carried_loads`` are each rank's loads from earlier buckets."""
    if hosts is not None:
        owners = assign_to_hosts_then_ranks(
            [param.numel() for _fqn, param in named_params],
            hosts,
            carried_loads=carried_loads,
        )
        return {fqn: owner for (fqn, _param), owner in zip(named_params, owners)}
    carried = [0] * world_size if carried_loads is None else list(carried_loads)
    loads = [(carried[rank], rank) for rank in range(world_size)]
    heapq.heapify(loads)
    assignments: dict[str, int] = {}
    for fqn, param in sorted(
        named_params,
        key=lambda item: item[1].numel(),
        reverse=True,
    ):
        load, rank = heapq.heappop(loads)
        assignments[fqn] = rank
        heapq.heappush(loads, (load + param.numel(), rank))
    return assignments


def make_bucketed_owned_full_param_segments(
    named_params: list[tuple[str, nn.Parameter]],
    world_size: int,
    *,
    hosts: Sequence[int] | None = None,
    carried_loads: Sequence[int] | None = None,
) -> dict[str, list[BucketedOwnedSegmentSpec]]:
    """Build one whole-parameter BucketedOwned segment per parameter.

    Parameters are assigned to owners with greedy LPT by numel; with ``hosts``
    (each rank's host, see ``mesh_rank_hosts``), hosts first. ``carried_loads``
    are each rank's owned numel from earlier buckets, which breaks ties.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, but got {world_size}.")
    if hosts is not None and len(hosts) != world_size:
        raise ValueError(f"hosts must give each of {world_size} ranks a host.")
    if carried_loads is not None and len(carried_loads) != world_size:
        raise ValueError(f"carried_loads must give each of {world_size} ranks a load.")
    assignments = _assign_params_to_ranks(
        named_params, world_size, hosts, carried_loads
    )
    return {
        fqn: [
            BucketedOwnedSegmentSpec(
                name=f"{fqn}#full",
                fqn=fqn,
                param_offset=0,
                numel=param.numel(),
                owner_rank=assignments[fqn],
                storage_order=param_order,
            )
        ]
        for param_order, (fqn, param) in enumerate(named_params)
    }


def make_bucketed_owned_full_param_placement_fn(
    *, carry_loads: bool = False
) -> PlacementFn:
    """Return a whole-parameter BucketedOwned placement function. Owners
    are balanced over the mesh's hosts first (``mesh_rank_hosts``), so heavy
    and light hosts alternate in rank order.

    With ``carry_loads``, the function keeps each mesh's per-rank owned numel
    across calls, so a bucket's owners also balance against the buckets placed
    before it, e.g. one owner bucket per transformer block, which would
    otherwise all get the same owners. Every rank must then call it for the
    same buckets in the same order, as SPMD code does."""
    carried: dict[tuple[int, ...], list[int]] = {}

    def placement_fn(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        mesh_ranks = tuple(mesh.mesh.flatten().tolist())
        loads = carried.setdefault(mesh_ranks, [0] * mesh.size())
        segments_by_fqn = make_bucketed_owned_full_param_segments(
            named_params,
            mesh.size(),
            hosts=mesh_rank_hosts(list(mesh_ranks)),
            carried_loads=loads if carry_loads else None,
        )
        if carry_loads:
            for segments in segments_by_fqn.values():
                for segment in segments:
                    loads[segment.owner_rank] += segment.numel
        placement = BucketedOwned(segments_by_fqn, view_kind="full_param")
        return {fqn: (placement,) for fqn, _ in named_params}

    return placement_fn


def _expert_block_order_key(fqn: str, suffix_order: tuple[str, ...]) -> int:
    for index, suffix in enumerate(suffix_order):
        if fqn.endswith(suffix):
            return index
    raise ValueError(f"Unexpected grouped expert weight FQN: {fqn}")


def make_bucketed_owned_expert_block_segments(
    named_params: list[tuple[str, nn.Parameter]],
    world_size: int,
    *,
    suffix_order: tuple[str, ...] = (".w1", ".w3", ".w2"),
) -> dict[str, list[BucketedOwnedSegmentSpec]]:
    """Build per-expert BucketedOwned segments for packed expert weights.

    Each weight stores experts along dim 0 and has at least 3 dims. Experts are
    assigned to contiguous owner ranges. ``suffix_order`` preserves the
    declarative expert-matrix order; lowering uses per-parameter BlockShard
    layouts.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, but got {world_size}.")
    if not suffix_order:
        raise ValueError("suffix_order must contain at least one suffix.")
    if not named_params:
        return {}

    bad = [
        (fqn, tuple(param.shape))
        for fqn, param in named_params
        if param.dim() < 3 or not fqn.endswith(suffix_order)
    ]
    if bad:
        raise ValueError(
            f"BucketedOwned expert block expects packed weights with at least 3 "
            f"dims and a suffix in {suffix_order}: {bad}"
        )

    num_experts = named_params[0][1].shape[0]
    if any(param.shape[0] != num_experts for _, param in named_params):
        raise ValueError("BucketedOwned expert block requires matching expert counts.")

    ordered_params = sorted(
        named_params,
        key=lambda item: _expert_block_order_key(item[0], suffix_order),
    )
    segments_by_fqn: dict[str, list[BucketedOwnedSegmentSpec]] = {
        fqn: [] for fqn, _ in ordered_params
    }
    experts_per_owner = max(1, (num_experts + world_size - 1) // world_size)
    for expert_idx in range(num_experts):
        owner = min(expert_idx // experts_per_owner, world_size - 1)
        for param_order, (fqn, param) in enumerate(ordered_params):
            expert_numel = math.prod(param.shape[1:])
            segments_by_fqn[fqn].append(
                BucketedOwnedSegmentSpec(
                    name=f"{fqn}#expert{expert_idx}",
                    fqn=fqn,
                    param_offset=expert_idx * expert_numel,
                    numel=expert_numel,
                    owner_rank=owner,
                    storage_order=expert_idx * len(ordered_params) + param_order,
                )
            )
    return segments_by_fqn


def make_bucketed_owned_expert_block_placement_fn(
    *,
    suffix_order: tuple[str, ...] = (".w1", ".w3", ".w2"),
) -> PlacementFn:
    """Return a placement function for one packed expert-weight bucket."""

    def placement_fn(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        segments_by_fqn = make_bucketed_owned_expert_block_segments(
            named_params,
            mesh.size(),
            suffix_order=suffix_order,
        )
        placement = BucketedOwned(segments_by_fqn, view_kind="expert_block")
        return {fqn: (placement,) for fqn, _ in named_params}

    return placement_fn


__all__ = [
    "BucketedOwned",
    "BucketedOwnedSegmentSpec",
    "make_bucketed_owned_expert_block_placement_fn",
    "make_bucketed_owned_expert_block_segments",
    "make_bucketed_owned_full_param_placement_fn",
    "make_bucketed_owned_full_param_segments",
]
