# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import itertools
import logging
import weakref

import torch
import torch.distributed as dist
from torch._logging import warning_once


logger = logging.getLogger(__name__)

_P2PTopology = tuple[int, ...]
_P2PWarmupRound = tuple[tuple[int, int], ...]
_P2PGroupCacheKey = tuple[_P2PTopology, str]
_P2PGroupCacheEntry = tuple[
    dict[tuple[int, int], dist.ProcessGroup], tuple[_P2PWarmupRound, ...]
]
_PP_EDGE_GROUP_CACHE: weakref.WeakKeyDictionary[
    dist.ProcessGroup, dict[_P2PGroupCacheKey, _P2PGroupCacheEntry]
] = weakref.WeakKeyDictionary()


def _p2p_topology(
    stage_index_to_group_rank: dict[int, int], group_size: int
) -> _P2PTopology:
    """Validate and return the physical-rank assignment for logical stages."""
    stage_indices = set(stage_index_to_group_rank)
    expected_indices = set(range(len(stage_index_to_group_rank)))
    if stage_indices != expected_indices:
        raise ValueError(
            "Pipeline stage mapping must contain contiguous indices starting at 0"
        )
    topology = tuple(
        stage_index_to_group_rank[stage_index]
        for stage_index in range(len(expected_indices))
    )
    if any(rank < 0 or rank >= group_size for rank in topology):
        raise ValueError(
            f"Pipeline stage mapping contains a rank outside [0, {group_size})"
        )
    return topology


def _p2p_edge_matchings(topology: _P2PTopology) -> tuple[_P2PWarmupRound, ...]:
    """Partition used physical edges into deterministic directed matchings."""
    remaining = sorted(
        {
            (min(source, destination), max(source, destination))
            for source, destination in itertools.pairwise(topology)
            if source != destination
        }
    )
    matchings: list[tuple[tuple[int, int], ...]] = []
    while remaining:
        used_ranks: set[int] = set()
        matching: list[tuple[int, int]] = []
        deferred: list[tuple[int, int]] = []
        for edge in remaining:
            if edge[0] in used_ranks or edge[1] in used_ranks:
                deferred.append(edge)
                continue
            matching.append(edge)
            used_ranks.update(edge)
        matchings.append(tuple(matching))
        remaining = deferred

    rounds: list[_P2PWarmupRound] = []
    for matching_edges in matchings:
        rounds.append(matching_edges)
        rounds.append(
            tuple((destination, source) for source, destination in matching_edges)
        )
    return tuple(rounds)


def _warn_if_eager_nccl(group: dist.ProcessGroup | None) -> None:
    """Warn when pipeline P2P uses an eagerly initialized NCCL communicator."""
    if dist.get_backend(group) not in {"nccl", "nccl2"}:
        return
    warning_once(
        logger,
        "Pipeline parallelism is using an eager NCCL communicator. Consider "
        'creating its process group with backend="nccl-lazy" so peer '
        "communicators are initialized lazily and traffic to different peers "
        "can overlap.",
    )


def _build_p2p_edge_groups(
    group: dist.ProcessGroup | None,
    stage_index_to_group_rank: dict[int, int],
    device: torch.device,
) -> tuple[dict[tuple[int, int], dist.ProcessGroup], tuple[_P2PWarmupRound, ...]]:
    """Create communicators for the directed rank edges used by a schedule."""
    parent = group if group is not None else dist.distributed_c10d._get_default_group()
    group_size = dist.get_world_size(parent)
    topology = _p2p_topology(stage_index_to_group_rank, group_size)
    cache_key = (topology, device.type)
    cached = _PP_EDGE_GROUP_CACHE.get(parent, {}).get(cache_key)
    if cached is not None:
        return cached

    group_rank = dist.get_rank(parent)
    rounds = _p2p_edge_matchings(topology)
    groups: dict[tuple[int, int], dist.ProcessGroup] = {}
    if str(dist.get_backend(parent)) == "fake":
        for round_edges in rounds:
            for edge in round_edges:
                if group_rank in edge:
                    groups[edge] = parent
        _PP_EDGE_GROUP_CACHE.setdefault(parent, {})[cache_key] = (groups, rounds)
        return groups, rounds

    # Match split_group's backend selection so its default-backend validation
    # sees the same device/backend mapping as this optional filter.
    parent_backend_config = dist.BackendConfig(dist.get_backend_config(parent))
    device_backend_map = parent_backend_config.get_device_backend_map()
    if device.type not in device_backend_map:
        raise ValueError(
            f"Pipeline process group has no backend for device type {device.type!r}"
        )
    backend_name = device_backend_map[device.type]
    backend_filter = None
    if len(set(device_backend_map.values())) > 1:
        candidate = f"{device.type}:{backend_name}"
        parent_default_type = (
            dist.distributed_c10d._get_default_backend_type_for_backend_config(
                parent_backend_config, parent.bound_device_id
            )
        )
        candidate_default_type = (
            dist.distributed_c10d._get_default_backend_type_for_backend_config(
                dist.BackendConfig(candidate), parent.bound_device_id
            )
        )
        if candidate_default_type == parent_default_type:
            backend_filter = candidate

    use_torchcomms = dist.distributed_c10d._use_torchcomms_enabled()
    timeout = None
    if not use_torchcomms:
        parent_backend = parent._get_backend(device)
        if not parent_backend.supports_splitting:
            raise RuntimeError(
                f"Pipeline P2P edge groups require backend {backend_name!r} to "
                "support split_group"
            )
        # Preserve the timeout configured on this exact native backend.
        timeout = parent_backend.options._timeout
    for round_index, round_edges in enumerate(rounds):
        child = dist.split_group(
            parent_pg=parent,
            split_ranks=[list(edge) for edge in round_edges],
            group_desc=f"pp_p2p_round_{round_index}",
            backend=backend_filter,
            timeout=timeout,
        )
        local_edge = next((edge for edge in round_edges if group_rank in edge), None)
        if local_edge is not None:
            if not isinstance(child, dist.ProcessGroup):
                raise AssertionError(
                    f"expected process group for edge {local_edge}, got {type(child)}"
                )
            groups[local_edge] = child

    logger.info(
        "Pipeline P2P: using %d directed rank-edge split rounds",
        len(rounds),
    )
    _PP_EDGE_GROUP_CACHE.setdefault(parent, {})[cache_key] = (groups, rounds)
    return groups, rounds


def _warmup_p2p_edge_groups(
    parent: dist.ProcessGroup,
    groups: dict[tuple[int, int], dist.ProcessGroup],
    rounds: tuple[_P2PWarmupRound, ...],
    device: torch.device,
) -> None:
    """Exercise each directed child communicator before graph capture."""
    group_rank = dist.get_rank(parent)
    # Matching edges are disjoint. Waiting before the next round lets every
    # round safely reuse this one setup-only buffer.
    tensor = torch.zeros(1, dtype=torch.int32, device=device)
    for round_edges in rounds:
        local_edge = next((edge for edge in round_edges if group_rank in edge), None)
        if local_edge is None:
            continue
        source, destination = local_edge
        op = dist.P2POp(
            dist.isend if group_rank == source else dist.irecv,
            tensor,
            dist.get_global_rank(
                parent, destination if group_rank == source else source
            ),
            groups[local_edge],
        )
        for work in dist.batch_isend_irecv([op]):
            work.wait()
