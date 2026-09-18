# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import itertools
import logging
import weakref
from collections.abc import Mapping

import torch
import torch.distributed as dist
from torch._logging import warning_once


logger = logging.getLogger(__name__)

_StageRankAssignment = tuple[int, ...]
_DirectedRankEdge = tuple[int, int]
_P2PSplitRound = tuple[_DirectedRankEdge, ...]
_DirectedP2PGroupMap = dict[_DirectedRankEdge, dist.ProcessGroup]
_P2PGroupCacheKey = tuple[_StageRankAssignment, str]
_P2PGroupCacheEntry = tuple[_DirectedP2PGroupMap, tuple[_P2PSplitRound, ...]]
# A schedule may configure several local stage objects for one assignment. Cache
# their shared children so every parent rank executes each split collective
# exactly once. Full process-group teardown owns the real child lifetimes.
_PP_EDGE_GROUP_CACHE: weakref.WeakKeyDictionary[
    dist.ProcessGroup, dict[_P2PGroupCacheKey, _P2PGroupCacheEntry]
] = weakref.WeakKeyDictionary()


def _stage_rank_assignment(
    stage_index_to_group_rank: dict[int, int], group_size: int
) -> _StageRankAssignment:
    """Return the physical pipeline-group rank assigned to each logical stage.

    Args:
        stage_index_to_group_rank: Mapping from each logical stage index to its
            rank in the pipeline process group.
        group_size: Number of ranks in the pipeline process group.

    Returns:
        A tuple indexed by logical stage, whose values are pipeline-group ranks.

    Raises:
        ValueError: If stage indices are not contiguous from zero or an assigned
            rank falls outside the pipeline process group.
    """
    stage_indices = set(stage_index_to_group_rank)
    expected_indices = set(range(len(stage_index_to_group_rank)))
    if stage_indices != expected_indices:
        raise ValueError(
            "Pipeline stage mapping must contain contiguous indices starting at 0"
        )
    assignment = tuple(
        stage_index_to_group_rank[stage_index]
        for stage_index in range(len(expected_indices))
    )
    if any(rank < 0 or rank >= group_size for rank in assignment):
        raise ValueError(
            f"Pipeline stage mapping contains a rank outside [0, {group_size})"
        )
    return assignment


def _directed_edge_split_rounds(
    stage_rank_assignment: _StageRankAssignment,
) -> tuple[_P2PSplitRound, ...]:
    """Partition adjacent physical-rank edges into collective split rounds.

    Logical adjacencies mapped to the same physical-rank pair share a child
    communicator. Same-rank adjacencies require no communication. Every
    remaining rank pair appears in two directed rounds. Each returned round is
    passed directly as one ``split_ranks`` argument to
    :func:`torch.distributed.split_group`.
    Because a parent rank may occur in at most one subgroup per split call, the
    helper packs only disjoint edges into a round. Its deterministic greedy
    matching combines independent edges and thereby reduces collective setup
    calls without changing which directed communicators are created.

    For example, the PP4/VPP2 assignment ``(0, 1, 2, 3, 0, 1, 2, 3)`` needs
    eight directed communicators. They fit in four split calls::

        ((0, 1), (2, 3))
        ((1, 0), (3, 2))
        ((0, 3), (1, 2))
        ((3, 0), (2, 1))

    Only successive logical stages in ``stage_rank_assignment`` contribute
    edges. Long-range skip connections are not supported; a future arbitrary
    connector set would need to be derived from the finalized schedule actions
    rather than from stage placement alone.

    Args:
        stage_rank_assignment: Physical pipeline-group rank for every logical
            stage, indexed by logical stage.

    Returns:
        Ordered split rounds containing directed ``(source_rank,
        destination_rank)`` edges.
    """
    remaining_edges = sorted(
        {
            (min(source, destination), max(source, destination))
            for source, destination in itertools.pairwise(stage_rank_assignment)
            if source != destination
        }
    )
    edge_matchings: list[tuple[_DirectedRankEdge, ...]] = []
    while remaining_edges:
        used_ranks: set[int] = set()
        matching: list[_DirectedRankEdge] = []
        deferred: list[_DirectedRankEdge] = []
        for edge in remaining_edges:
            if edge[0] in used_ranks or edge[1] in used_ranks:
                deferred.append(edge)
                continue
            matching.append(edge)
            used_ranks.update(edge)
        edge_matchings.append(tuple(matching))
        remaining_edges = deferred

    split_rounds: list[_P2PSplitRound] = []
    for matching_edges in edge_matchings:
        split_rounds.append(matching_edges)
        split_rounds.append(
            tuple((destination, source) for source, destination in matching_edges)
        )
    return tuple(split_rounds)


def _warn_if_eager_nccl(group: dist.ProcessGroup | None) -> None:
    """Warn when the shared pipeline communicator eagerly initializes NCCL.

    Args:
        group: Pipeline process group, or ``None`` for the default group.
    """
    if dist.get_backend(group) not in {"nccl", "nccl2"}:
        return
    warning_once(
        logger,
        "Pipeline parallelism is using an eager NCCL communicator. Consider "
        'creating its process group with backend="nccl-lazy" so peer '
        "communicators are initialized lazily and traffic to different peers "
        "can overlap.",
    )


def _initialize_additional_parent_backends(
    parent: dist.ProcessGroup,
    device_backend_map: Mapping[str, str],
    initialized_backend: str,
) -> None:
    """Initialize retained native backends not exercised on the stage device.

    The schedule initializes the backend for the stage device before reaching
    this boundary. An unfiltered mixed-backend child retains every distinct
    backend, and ``split_group`` requires each retained backend to be initialized.

    Args:
        parent: Parent pipeline process group.
        device_backend_map: Parent mapping from device type to backend name.
        initialized_backend: Backend already initialized on the stage device.
    """
    initialized = {initialized_backend}
    for device_type, backend_name in sorted(device_backend_map.items()):
        if backend_name in initialized:
            continue
        backend_device = torch.device(device_type)
        bound_device = parent.bound_device_id
        if bound_device is not None and bound_device.type == device_type:
            backend_device = bound_device
        dist.all_reduce(
            torch.zeros(1, dtype=torch.int32, device=backend_device),
            group=parent,
        )
        initialized.add(backend_name)


def _build_p2p_edge_groups(
    group: dist.ProcessGroup | None,
    stage_index_to_group_rank: dict[int, int],
    device: torch.device,
) -> tuple[_DirectedP2PGroupMap, tuple[_P2PSplitRound, ...]]:
    """Create or reuse child groups for a schedule's directed rank edges.

    Child groups are keyed by ``(source_group_rank, destination_group_rank)``.
    Opposite directions use distinct keys and communicators, while repeated
    logical-stage edges mapped to the same directed physical edge share one
    communicator FIFO. Only groups incident to the calling rank are returned.

    Args:
        group: Parent pipeline process group, or ``None`` for the default group.
        stage_index_to_group_rank: Mapping from logical stage index to rank in
            the parent pipeline process group.
        device: Device used by this rank's pipeline stages.

    Returns:
        The local directed-edge group map and the deterministic split rounds
        that every parent rank must execute in order.

    Raises:
        ValueError: If the stage assignment is invalid or the parent has no
            backend for ``device``.
        RuntimeError: If the selected native backend cannot split groups.
    """
    parent = group if group is not None else dist.distributed_c10d._get_default_group()
    group_size = dist.get_world_size(parent)
    stage_rank_assignment = _stage_rank_assignment(
        stage_index_to_group_rank, group_size
    )
    cache_key = (stage_rank_assignment, device.type)
    cached = _PP_EDGE_GROUP_CACHE.get(parent, {}).get(cache_key)
    if cached is not None:
        return cached

    group_rank = dist.get_rank(parent)
    split_rounds = _directed_edge_split_rounds(stage_rank_assignment)
    groups: _DirectedP2PGroupMap = {}
    if str(dist.get_backend(parent)) == "fake":
        for round_edges in split_rounds:
            for edge in round_edges:
                if group_rank in edge:
                    groups[edge] = parent
        return groups, split_rounds

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
        if backend_filter is None and len(set(device_backend_map.values())) > 1:
            _initialize_additional_parent_backends(
                parent,
                device_backend_map,
                backend_name,
            )
    for round_index, round_edges in enumerate(split_rounds):
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
        len(split_rounds),
    )
    _PP_EDGE_GROUP_CACHE.setdefault(parent, {})[cache_key] = (groups, split_rounds)
    return groups, split_rounds


def _preconnect_p2p_edge_groups(
    parent: dist.ProcessGroup,
    groups: _DirectedP2PGroupMap,
    split_rounds: tuple[_P2PSplitRound, ...],
    device: torch.device,
) -> None:
    """Exercise every local directed child P2P path before execution.

    ``split_group`` has already created each child communicator at this point.
    This setup-only exchange preconnects the actual send/receive paths in
    deterministic rounds so first use cannot occur during pipeline execution or
    CUDA-graph capture. It is distinct from a runtime's model warmup.

    Args:
        parent: Parent pipeline process group used to translate group ranks to
            global ranks.
        groups: Local child groups keyed by directed
            ``(source_group_rank, destination_group_rank)`` edges.
        split_rounds: Deterministic directed rounds returned by
            ``_build_p2p_edge_groups``.
        device: Device on which to allocate the setup-only payload.
    """
    group_rank = dist.get_rank(parent)
    # Matching edges are disjoint. Waiting before the next round lets every
    # round safely reuse this one setup-only buffer.
    tensor = torch.zeros(1, dtype=torch.int32, device=device)
    for round_edges in split_rounds:
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
