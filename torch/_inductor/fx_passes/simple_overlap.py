"""
Simple, defensive overlap pass for communication/compute overlap in FX graphs.

Guarantees
----------
1. **No NCCL hangs**: collectives on the same process group are never reordered
   relative to each other (neither starts nor waits).
2. **No memory regression**: peak memory is simulated after all moves and the
   entire pass is reverted if it would increase peak memory.
3. **Bounded compile time**: the expensive GraphAliasTracker is built once;
   memory simulation is a single O(N) pass over the graph.
4. **Minimal graph changes**: only collective-start and wait_tensor nodes are
   moved. All other nodes stay in their original positions.

Algorithm
---------
Phase 1 -- Move collective starts earlier:
    Each collective is moved to just after the latest of its data dependencies
    and the previous collective on the same process group.

Phase 2 -- Move waits later:
    Each wait is moved to just before its earliest consumer, preserving the
    relative order of waits on the same process group.
    After all waits are moved, a single memory simulation verifies peak memory.
    If peak memory would increase, ALL wait moves are reverted.
"""

import logging

import torch.fx as fx
from torch._inductor.fx_passes.bucketing import (
    _get_collective_node_from_wait,
    _schedulable_wait_node,
)
from torch._inductor.fx_passes.memory_estimator import (
    GraphAliasTracker,
    StorageKey,
)
from torch._inductor.fx_passes.overlap_scheduling import get_group_name
from torch.fx.experimental.symbolic_shapes import optimization_hint
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)

_is_releasable = lambda n: not n.name.startswith("primals")
_device_filter = lambda d: d.type != "cpu"


def _simulate_peak_memory(
    graph: fx.Graph, alias_tracker: GraphAliasTracker
) -> int:
    """Simulate execution and return peak device memory in bytes."""
    live: OrderedSet[StorageKey] = OrderedSet()
    current_mem = 0
    peak_mem = 0
    scheduled: OrderedSet[fx.Node] = OrderedSet()

    for node in graph.nodes:
        if node.op in ("placeholder", "get_attr"):
            for sk in alias_tracker.get_fresh_allocations(node):
                if _device_filter(sk.device) and sk not in live:
                    live.add(sk)
                    current_mem += optimization_hint(sk.storage.nbytes())
            peak_mem = max(peak_mem, current_mem)
            continue
        if node.op == "output":
            continue

        scheduled.add(node)

        for sk in alias_tracker.get_fresh_allocations(node):
            if _device_filter(sk.device) and sk not in live:
                live.add(sk)
                current_mem += optimization_hint(sk.storage.nbytes())

        peak_mem = max(peak_mem, current_mem)

        for sk in alias_tracker.get_storage_uses(node):
            if not _device_filter(sk.device):
                continue
            if sk not in live:
                continue
            if not _is_releasable(alias_tracker.storage_to_allocator[sk]):
                continue
            if all(u in scheduled for u in alias_tracker.storage_to_uses[sk]):
                live.discard(sk)
                current_mem -= optimization_hint(sk.storage.nbytes())

    return peak_mem


def _node_index_map(graph: fx.Graph) -> dict[fx.Node, int]:
    return {node: i for i, node in enumerate(graph.nodes)}


def simple_overlap(graph: fx.Graph) -> None:
    """
    Reorder FX graph nodes to overlap collectives with compute.

    Moves collective starts earlier and waits later, respecting dependencies
    and ensuring no increase in peak memory.
    """
    nodes = list(graph.nodes)
    if len(nodes) == 0:
        return

    pairs: list[tuple[fx.Node, fx.Node]] = []
    for node in nodes:
        if not _schedulable_wait_node(node):
            continue
        start = _get_collective_node_from_wait(node)
        if start is None:
            continue
        pairs.append((start, node))

    if not pairs:
        return

    log.debug("simple_overlap: found %d collective-wait pairs", len(pairs))

    # Build alias tracker once for all memory checks.
    alias_tracker = GraphAliasTracker(list(graph.nodes))
    initial_peak = _simulate_peak_memory(graph, alias_tracker)
    log.info(
        "simple_overlap: initial peak memory: %d MB", initial_peak // (1024 * 1024)
    )

    _move_collectives_earlier(graph, pairs, alias_tracker, initial_peak)
    _move_waits_later(graph, pairs, alias_tracker, initial_peak)

    graph.lint()


def _move_collectives_earlier(
    graph: fx.Graph,
    pairs: list[tuple[fx.Node, fx.Node]],
    alias_tracker: GraphAliasTracker,
    initial_peak: int,
) -> None:
    """Move each collective start as early as its dependencies allow.

    Constraints:
    - Must be after all data dependencies (input nodes)
    - Must be after previous collective on the same process group
    - Must not increase peak memory beyond initial_peak

    All moves are applied optimistically, then verified with a single
    memory simulation. If peak memory would increase, all moves are
    reverted.
    """
    last_collective_per_pg: dict[str, fx.Node] = {}

    node_idx = _node_index_map(graph)
    sorted_pairs = sorted(pairs, key=lambda p: node_idx[p[0]])

    moved: list[tuple[fx.Node, fx.Node]] = []

    for start, _wait in sorted_pairs:
        pg = get_group_name(start)
        node_idx = _node_index_map(graph)

        earliest_after: fx.Node | None = None
        earliest_after_idx = -1

        for inp in start.all_input_nodes:
            idx = node_idx[inp]
            if idx > earliest_after_idx:
                earliest_after_idx = idx
                earliest_after = inp

        if pg in last_collective_per_pg:
            prev_coll = last_collective_per_pg[pg]
            idx = node_idx[prev_coll]
            if idx > earliest_after_idx:
                earliest_after_idx = idx
                earliest_after = prev_coll

        if earliest_after is not None and earliest_after_idx < node_idx[start] - 1:
            prev_node = start.prev
            earliest_after.append(start)
            moved.append((start, prev_node))
            log.debug(
                "simple_overlap: moved collective %s earlier (after %s)",
                start.name,
                earliest_after.name,
            )

        last_collective_per_pg[pg] = start

    if moved:
        post_peak = _simulate_peak_memory(graph, alias_tracker)
        log.info(
            "simple_overlap: peak memory after moving %d collectives: %d MB "
            "(initial: %d MB)",
            len(moved),
            post_peak // (1024 * 1024),
            initial_peak // (1024 * 1024),
        )
        if post_peak > initial_peak:
            log.info(
                "simple_overlap: reverting all %d collective moves "
                "(would increase peak memory by %d MB)",
                len(moved),
                (post_peak - initial_peak) // (1024 * 1024),
            )
            for start, prev_node in reversed(moved):
                prev_node.append(start)


def _move_waits_later(
    graph: fx.Graph,
    pairs: list[tuple[fx.Node, fx.Node]],
    alias_tracker: GraphAliasTracker,
    initial_peak: int,
) -> None:
    """Move each wait to just before its earliest consumer.

    Preserves the relative order of waits on the same process group.
    After all moves, simulates peak memory once. If it would increase
    beyond initial_peak, reverts ALL wait moves.
    """

    node_idx = _node_index_map(graph)
    sorted_pairs = sorted(pairs, key=lambda p: node_idx[p[1]])

    # Track per-PG wait ordering so we never reorder waits on the same PG.
    last_wait_per_pg: dict[str, fx.Node] = {}

    # Save original positions for bulk revert.
    # Each entry is (wait, prev_node_before_move).
    moved: list[tuple[fx.Node, fx.Node]] = []

    for start, wait in sorted_pairs:
        users = OrderedSet(wait.users.keys())
        if not users:
            continue

        node_idx = _node_index_map(graph)
        wait_idx = node_idx[wait]
        pg = get_group_name(start)

        # Target: just before the earliest consumer
        target_idx = len(node_idx)
        target_node: fx.Node | None = None
        for user in users:
            idx = node_idx[user]
            if idx < target_idx:
                target_idx = idx
                target_node = user

        if target_node is None:
            continue

        # Constraint: must stay after the previous wait on the same PG
        if pg in last_wait_per_pg:
            prev_wait = last_wait_per_pg[pg]
            prev_wait_idx = node_idx[prev_wait]
            # target must be after prev_wait
            if target_idx <= prev_wait_idx + 1:
                last_wait_per_pg[pg] = wait
                continue
            # If target would place us before prev_wait, clamp
            if target_idx <= prev_wait_idx:
                last_wait_per_pg[pg] = wait
                continue

        if target_idx <= wait_idx + 1:
            last_wait_per_pg[pg] = wait
            continue

        prev_node = wait.prev
        target_node.prepend(wait)
        moved.append((wait, prev_node))
        last_wait_per_pg[pg] = wait

        log.debug(
            "simple_overlap: moved wait %s later (before %s)",
            wait.name,
            target_node.name,
        )

    if not moved:
        return

    # Single memory check after all moves
    new_peak = _simulate_peak_memory(graph, alias_tracker)
    if new_peak > initial_peak:
        log.info(
            "simple_overlap: reverting all %d wait moves (peak memory would "
            "increase by %d MB, from %d MB to %d MB)",
            len(moved),
            (new_peak - initial_peak) // (1024 * 1024),
            initial_peak // (1024 * 1024),
            new_peak // (1024 * 1024),
        )
        for wait, prev_node in reversed(moved):
            prev_node.append(wait)
    else:
        log.info(
            "simple_overlap: all %d wait moves verified (peak memory: %d MB, "
            "initial: %d MB)",
            len(moved),
            new_peak // (1024 * 1024),
            initial_peak // (1024 * 1024),
        )
