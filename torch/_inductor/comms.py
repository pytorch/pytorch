# mypy: allow-untyped-defs
# pyre-strict
from __future__ import annotations

import heapq
import importlib
import itertools
import logging
import operator
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
from torch._logging import trace_structured
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._ordered_set import OrderedSet

from . import config, config_comms, ir
from .dependencies import WeakDep


if TYPE_CHECKING:
    from .ir import IRNode, Operation

from .memory import (
    estimate_peak_memory_allocfree,
    FreeableInputBuffer,
    get_freeable_input_buf,
    SNodeMemory,
)
from .utils import (
    contains_collective,
    contains_wait,
    find_recursive_deps_of_node,
    find_recursive_users_of_node,
    is_collective,
    is_fallback_op,
    is_wait,
)
from .virtualized import V


log = logging.getLogger(__name__)
overlap_log = torch._logging.getArtifactLogger(__name__, "overlap")

if TYPE_CHECKING:
    from torch._inductor.scheduler import BaseSchedulerNode


def align_runtime_estimations_across_all_distributed_ranks(
    snodes: list[BaseSchedulerNode],
):
    from torch._inductor.scheduler import _get_mm_like_fn

    runtime_estimations = {}
    runtime_estimations_for_mms = {}

    for snode in snodes:
        runtime_estimations[snode] = snode.get_estimated_runtime()
        if _get_mm_like_fn(snode) is not None:
            runtime_estimations_for_mms[snode] = runtime_estimations[snode]

    import torch.distributed as dist
    from torch.distributed.distributed_c10d import _get_default_group

    world_size = dist.get_world_size()
    pg = _get_default_group()
    gathered_runtime_estimations_for_mms: list[list[float]] = [
        [] for _ in range(world_size)
    ]
    dist.all_gather_object(
        gathered_runtime_estimations_for_mms,
        list(runtime_estimations_for_mms.values()),
        pg,
    )
    median_runtime_estimations_for_mms = torch.median(
        torch.tensor(gathered_runtime_estimations_for_mms), dim=0
    ).values.tolist()
    for idx, snode in enumerate(runtime_estimations_for_mms.keys()):
        runtime_estimations_for_mms[snode] = median_runtime_estimations_for_mms[idx]

    for snode in snodes:
        if snode in runtime_estimations_for_mms:
            runtime_estimations[snode] = runtime_estimations_for_mms[snode]
        snode.override_estimated_runtime = runtime_estimations[snode]


def sink_waits(snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Greedily schedules waits as late as possible.
    """
    return _schedule_for_comm(
        snodes, raise_comms=False, sink_waits=True, reorder_for_overlap=False
    )


def raise_comms(snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Greedily schedules comms as early as possible.
    """
    return _schedule_for_comm(
        snodes, raise_comms=True, sink_waits=False, reorder_for_overlap=False
    )


def reorder_compute_for_overlap(
    snodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """
    This achieves the following overall scheduling procedure:
        Step 1: Given that we've currently scheduled comm N, we now schedule all compute nodes
            that are required for comm N + 1 but do not depend on comm N, to run at the same time with comm N.
        Step 2: If all those compute nodes are sufficient to overlap comm N, we're done.
            Otherwise, we now need to look elsewhere to find compute that overlaps with comm N.
            We prioritize compute nodes that are needed sooner.
        Step 3: We schedule the compute nodes dependent on comm N and required for comm N + 1.
        Step 4: We schedule comm N + 1.
        Repeat this for subsequent comm nodes.
    """
    return _schedule_for_comm(
        snodes, raise_comms=True, sink_waits=True, reorder_for_overlap=True
    )


def reorder_communication_preserving_peak_memory(
    snodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """
    Reorders communication ops relative to computation ops to improve communication-compute overlapping and hide comm
    latency.  Stops moving a particular op if it reaches a point that would have increased the peak memory footprint.

    Currently, follows these heuristics (subject to change or tune):
    - never reorders collectives relative to one another, for SPMD safety
    - has an option for per-collective prefetch limit, but does not enable it by default
    - limits the total number of reorder steps to some factor of the graph size to prevent worst-case quadratic
      performance

    Prerequisite: sink_comms_and_waits - ensure comm and wait nodes are scheduled as late as possible, respecting data
    dependencies.  That allows reorder_communication_preserving_peak_memory to take a best case peak-memory snapshot,
    and then monotonically improve latency by moving collectives backward in time.

    Peak memory impact is computed in an iterative fashion.  First, memory use at each timestep is computed, and global
    peak memory is computed as a max over timesteps.  Then, when swapping any two adjacent nodes, only the curr-memory
    for the earlier of the nodes after the swap is affected.  This enables checking step by step whether a swap is
    peak-memory-safe, and bailing out if not.  Example:

    0   n0      C0
    1   n1      C0 + Allocs(n1) - Frees(n1)
    2   n2      C0 + Allocs(n1) - Frees(n1) + Allocs(n2) - Frees(n2)

    0   n0      C0
    1   n2      C0 + Allocs(n2) - Frees(n2)    <-- After moving n2 to Time 1, only time1 memory changes
    2   n1      C0 + Allocs(n2) - Frees(n2) + Allocs(n1) - Frees(n1)

    """
    reordered_snodes, node_stats = (
        _reorder_communication_preserving_peak_memory_internal(snodes)
    )

    return reordered_snodes


@dataclass
class ReorderInfo:
    """
    Debug info describing how an individual snode was reordered
    """

    limiting_factor: str = "None"
    moves: int = 0
    grouped: int = 0
    grouped_info: str = ""
    comm_time: float = -1.0
    comp_time: float = -1.0
    initial_exposed: float = -1.0
    final_exposed: float = -1.0
    overlap_info: str = "None"

    @property
    def improvement(self):
        return self.initial_exposed - self.final_exposed


def is_gemm_like(node: IRNode | Operation | None) -> bool:
    if node is None:
        return False

    if is_fallback_op(
        node,  # type: ignore[arg-type]
        torch.ops.aten._scaled_dot_product_flash_attention.default,
    ):
        return True

    if (
        python_kernel_name := getattr(node, "python_kernel_name", None)
    ) and "extern_kernels" in python_kernel_name:
        return True
    return False


def contains_gemm_like(snode: BaseSchedulerNode) -> bool:
    from torch._inductor.scheduler import GroupedSchedulerNode

    if isinstance(snode, GroupedSchedulerNode):
        return any(contains_gemm_like(x) for x in snode.snodes)
    else:
        return is_gemm_like(snode.node)


def _temp_group_visit_leaves(snode: BaseSchedulerNode, fn):
    from torch._inductor.scheduler import GroupedSchedulerNode

    if isinstance(snode, GroupedSchedulerNode) and snode.temp_grouping:
        for _snode in snode.snodes:
            fn(_snode)
    else:
        fn(snode)


def _group_name(snode, with_bufs=False) -> str:
    ret = ""
    for n in snode.snodes:
        if ret:
            ret += "_"
        ret += n.get_name()
        if with_bufs:
            ret += f"{list(snode.get_buffer_names())}"
    return ret


def _is_fake_dep(d):
    return isinstance(d, WeakDep) and d.is_fake


def _group_names(gns: list[BaseSchedulerNode]) -> str:
    return "~".join([gn.get_name() for gn in gns])


def _initialize_memory_tracking(snodes, graph_inputs, graph_outputs):
    """Initialize memory tracking data structures"""
    name_to_freeable_input_buf = get_freeable_input_buf(snodes, graph_inputs)
    peak_memory, snodes_curr_memory, snodes_allocfree, buf_to_snode_last_use = (
        estimate_peak_memory_allocfree(
            snodes, name_to_freeable_input_buf, graph_outputs
        )
    )
    _curr_memory = dict(zip(snodes, snodes_curr_memory))
    # pyrefly: ignore [unsupported-operation]
    _curr_memory[None] = (0, 0)

    # Build candidate buffer map for optimization
    candidate_buffer_map = _build_candidate_buffer_map(buf_to_snode_last_use)

    return (
        peak_memory,
        _curr_memory,
        snodes_allocfree,
        buf_to_snode_last_use,
        name_to_freeable_input_buf,
        candidate_buffer_map,
    )


def _initialize_double_linked_list(
    snodes: list[BaseSchedulerNode],
) -> tuple[
    dict[BaseSchedulerNode, BaseSchedulerNode | None],
    dict[BaseSchedulerNode, BaseSchedulerNode | None],
    BaseSchedulerNode,
]:
    """Create double-linked list structure from snodes"""
    _prev = {}
    _next = {}
    for i, snode in enumerate(snodes):
        _prev[snode] = snodes[i - 1] if i > 0 else None
        _next[snode] = snodes[i + 1] if i < len(snodes) - 1 else None
    _head = snodes[0]
    return _prev, _next, _head


def _build_candidate_buffer_map(
    buf_to_snode_last_use: dict,
) -> dict[BaseSchedulerNode, OrderedSet]:
    """
    Build inverted index: node -> set of buffers where node appears in successors.

    This optimization reduces buffer iteration from O(total_buffers) to O(buffers_per_node).
    Since buffer successors are immutable during reordering, this map doesn't need updates.

    Returns:
        dict mapping each node to the set of buffers that have this node in their successors
    """
    node_to_candidate_bufs: dict[BaseSchedulerNode, OrderedSet] = defaultdict(
        OrderedSet
    )

    for buf in buf_to_snode_last_use:
        # Add to every successor node's buffer set
        for succ_node in buf.mpi_buffer.succ_nodes:
            node_to_candidate_bufs[succ_node].add(buf)

    return dict(node_to_candidate_bufs)


def _precompute_node_output_sets(
    snodes: list[BaseSchedulerNode],
) -> dict[BaseSchedulerNode, OrderedSet[str]]:
    """
    Pre-compute output name sets for all nodes.

    This optimization avoids creating OrderedSet objects repeatedly during
    exposed time calculations.

    Returns:
        dict mapping each node to a set of its output names
    """
    return {
        snode: OrderedSet(o.get_name() for o in snode.get_outputs()) for snode in snodes
    }


def _op_runtime_estimate_mult(snode):
    # Apply multipliers for faster experimentation.
    # TODO(ivankobzarev): Remove after confirmation that runtime estimations are correct.
    if contains_collective(snode):
        return config_comms.reorder_sink_runtime_estimations_comm_mult

    return config_comms.reorder_sink_runtime_estimations_non_comm_mult


def is_async_collective(snode):
    """
    Filtering out ops that contain Collective and Wait inside and considered as Collectives.
    See contains_collective function.
    If the op contains Wait inside - consider as Synchronous compute.
    """
    if python_kernel_name := getattr(snode.node, "python_kernel_name", None):
        if "torch.ops._dtensor.shard_dim_alltoall.default" in python_kernel_name:
            return False

    return True


def contains_async_collective(snode):
    return contains_collective(snode, is_async_collective)


def _group_nodes_from_linked_list(
    head: BaseSchedulerNode | None,
    tail: BaseSchedulerNode | None,
    next_dict: dict[BaseSchedulerNode, BaseSchedulerNode | None],
) -> list[BaseSchedulerNode]:
    """
    Traverse doubly-linked list from head to tail and return nodes as a list.

    Args:
        head: Starting node of the segment
        tail: Ending node of the segment (inclusive)
        next_dict: Dictionary mapping each node to its next node

    Returns:
        List of nodes from head to tail (inclusive)
    """
    ret = []
    n = head
    while True:
        if n is not None:
            ret.append(n)
        if n == tail:
            break
        n = next_dict[n]  # type: ignore[index]
    return ret


def _is_corresponding_collective_wait(
    collective_snode: BaseSchedulerNode,
    wait_snode: BaseSchedulerNode,
    node_output_sets: dict[BaseSchedulerNode, frozenset[str]],
    node_dep_sets: dict[BaseSchedulerNode, frozenset[str]],
) -> bool:
    """
    Check if a wait node corresponds to a given collective node.
    Uses pre-computed sets for O(1) lookup.
    """
    collective_outs = node_output_sets[collective_snode]
    unmet_deps = node_dep_sets[wait_snode]
    return bool(unmet_deps & collective_outs)


def _coll_exposed_communication_time(
    collective_snode: BaseSchedulerNode,
    next_dict: dict[BaseSchedulerNode, BaseSchedulerNode | None],
    runtimes: dict[BaseSchedulerNode, float],
    node_output_sets: dict[BaseSchedulerNode, frozenset[str]],
    node_dep_sets: dict[BaseSchedulerNode, frozenset[str]],
) -> tuple[float, float, str]:
    """
    Calculate exposed communication time by iterating directly over linked list.
    Avoids O(N) list construction for each call.

    The collective_snode is the starting point, iteration continues via next_dict.
    """
    comm_time = runtimes[collective_snode]
    comp_time = 0.0
    collective_outs = node_output_sets[collective_snode]
    overlap_info = ""
    collectives_found: list[BaseSchedulerNode] = []

    snode = next_dict[collective_snode]
    while snode is not None:
        unmet_deps = node_dep_sets[snode]

        if unmet_deps & collective_outs:
            overlap_info += f"->W[{snode.get_name()}]"
            break

        if contains_collective(snode):
            if not contains_async_collective(snode):
                break
            else:
                collectives_found.append(snode)
                snode = next_dict[snode]
                continue
        if contains_wait(snode):
            has_wait_for_collectives_found = False
            for _coll in collectives_found:
                if _is_corresponding_collective_wait(
                    collective_snode, snode, node_output_sets, node_dep_sets
                ):
                    has_wait_for_collectives_found = True
                    break
            if has_wait_for_collectives_found:
                break

        comp_time_before = comp_time

        def accumulate_time(_snode: BaseSchedulerNode) -> None:
            nonlocal comp_time
            comp_time += runtimes[_snode]

        _temp_group_visit_leaves(snode, accumulate_time)
        comp_time_after = comp_time
        overlap_info += f"+{snode.get_name()}[{comp_time_after - comp_time_before}]"

        snode = next_dict[snode]

    return comm_time, comp_time, overlap_info


def _wait_exposed_communication_time(
    wait_snode: BaseSchedulerNode,
    head: BaseSchedulerNode,
    prev_dict: dict[BaseSchedulerNode, BaseSchedulerNode | None],
    runtimes: dict[BaseSchedulerNode, float],
    node_output_sets: dict[BaseSchedulerNode, frozenset[str]],
    node_dep_sets: dict[BaseSchedulerNode, frozenset[str]],
) -> tuple[float, float, str]:
    """
    Calculate exposed communication time for a wait operation by iterating
    directly over linked list backwards. Avoids O(N) list construction.

    Iterates from wait_snode backwards using prev_dict to find corresponding collective.
    """
    comm_time = 0.0
    comp_time = 0.0
    overlap_info = ""
    waits_found: list[BaseSchedulerNode] = []

    snode = prev_dict[wait_snode]
    while snode is not None:
        if contains_wait(snode):
            waits_found.append(snode)
        if contains_collective(snode):
            if _is_corresponding_collective_wait(
                snode, wait_snode, node_output_sets, node_dep_sets
            ):
                comm_time = runtimes[snode]
                overlap_info += f"->C[{snode.get_name()}]"
                break

            if not contains_async_collective(snode):
                comp_time = 0.0
                snode = prev_dict[snode]
                continue
            else:
                for w in waits_found:
                    if _is_corresponding_collective_wait(
                        snode, w, node_output_sets, node_dep_sets
                    ):
                        comp_time = 0.0
                        break  # inner loop break
                snode = prev_dict[snode]
                continue

        comp_time_before = comp_time

        def accumulate_time(_snode: BaseSchedulerNode) -> None:
            nonlocal comp_time
            comp_time += runtimes[_snode]

        _temp_group_visit_leaves(snode, accumulate_time)
        comp_time_after = comp_time
        overlap_info += f"+{snode.get_name()}[{comp_time_after - comp_time_before}]"

        snode = prev_dict[snode]

    return comm_time, comp_time, overlap_info


def _perform_double_linked_list_swap(
    candidate: BaseSchedulerNode,
    group_head: BaseSchedulerNode,
    group_tail: BaseSchedulerNode,
    prev_dict: dict[BaseSchedulerNode, BaseSchedulerNode | None],
    next_dict: dict[BaseSchedulerNode, BaseSchedulerNode | None],
    head: BaseSchedulerNode,
) -> BaseSchedulerNode:
    """
    Swap positions of candidate and group in doubly-linked list.

    Transforms:
    candidate_prev -> candidate -> group_head...group_tail -> group_tail_next
    Into:
    candidate_prev -> group_head...group_tail -> candidate -> group_tail_next

    Args:
        candidate: Node to swap with group
        group_head: First node of group
        group_tail: Last node of group
        prev_dict: Dictionary mapping nodes to their previous nodes
        next_dict: Dictionary mapping nodes to their next nodes
        head: Current head of the linked list

    Returns:
        New head of the linked list (may change if candidate was the head)
    """
    # 0: Update candidate's previous node
    candidate_prev = prev_dict[candidate]
    if candidate_prev:
        next_dict[candidate_prev] = group_head
    prev_dict[group_head] = candidate_prev

    # 2: Update group_tail's next node
    group_tail_next = next_dict[group_tail]
    if group_tail_next:
        prev_dict[group_tail_next] = candidate
    next_dict[candidate] = group_tail_next

    # 1: Link group_tail to candidate
    prev_dict[candidate] = group_tail
    next_dict[group_tail] = candidate

    # Update head if candidate was the head
    if head == candidate:
        return group_head
    return head


def _calculate_potential_peak_memory_reorder(
    candidate: BaseSchedulerNode,
    gns: list[BaseSchedulerNode],
    group_tail: BaseSchedulerNode,
    group_peak_memory: int,
    candidate_delta_mem: int,
    candidate_allocfree: SNodeMemory,
    group_n_to_bufs_after_swap_dealloc_by_candidate: dict,
    curr_memory: dict,
) -> tuple[int, dict[BaseSchedulerNode, int]]:
    """
    Calculate potential peak memory after swapping candidate with group (reorder version).

    Computes new memory levels for all affected nodes and returns the potential
    peak memory along with cached post-allocation memory values for each node.

    Args:
        candidate: Node being moved
        gns: Group nodes
        group_tail: Last node of group
        group_peak_memory: Current peak memory within the group
        candidate_delta_mem: Net memory change from candidate (alloc - free)
        candidate_allocfree: Candidate's allocation/free info
        group_n_to_bufs_after_swap_dealloc_by_candidate: Buffers whose deallocation moves to candidate
        curr_memory: Current memory state dict

    Returns:
        Tuple of (potential_peak_memory, post_alloc_update_dict)
    """
    # Caching calculations of memory for group nodes and candidate,
    # to apply without recalculation after swap.
    _post_alloc_update: dict[BaseSchedulerNode, int] = {}
    potential_peak: int = 0
    if not group_n_to_bufs_after_swap_dealloc_by_candidate:
        # Not accounting for buffers last use change
        potential_peak = max(
            group_peak_memory - candidate_delta_mem,
            curr_memory[group_tail][1]
            - candidate_delta_mem
            + candidate_allocfree.size_alloc,
        )
        return potential_peak, _post_alloc_update

    # If candidate will be after group, the starting memory level of group nodes
    # changes to the -(candidate.size_alloc - candidate.size_free)
    mem_after_reorder_delta: int = -candidate_delta_mem
    for gn in gns:
        gn_post_alloc_mem = curr_memory[gn][0] + mem_after_reorder_delta
        _post_alloc_update[gn] = gn_post_alloc_mem
        potential_peak = max(potential_peak, gn_post_alloc_mem)

        bufs = group_n_to_bufs_after_swap_dealloc_by_candidate.get(gn)
        if bufs is not None:
            for buf in bufs:
                # Candidate will deallocate those buffers
                mem_after_reorder_delta += buf.mpi_buffer.size_free

    candidate_mem_post_alloc = (
        curr_memory[group_tail][1]
        + mem_after_reorder_delta
        + candidate_allocfree.size_alloc
    )
    _post_alloc_update[candidate] = candidate_mem_post_alloc
    potential_peak = max(potential_peak, candidate_mem_post_alloc)
    return potential_peak, _post_alloc_update


def _update_memory_tracking_after_swap_reorder(
    candidate: BaseSchedulerNode,
    gns: list[BaseSchedulerNode],
    group_tail: BaseSchedulerNode,
    candidate_delta_mem: int,
    candidate_allocfree: SNodeMemory,
    group_n_to_bufs_after_swap_dealloc_by_candidate: dict,
    post_alloc_update: dict[BaseSchedulerNode, int],
    curr_memory: dict,
    buf_to_snode_last_use: dict,
    snodes_allocfree: dict,
) -> None:
    """
    Update memory tracking structures after swap (reorder version).

    Updates curr_memory, buf_to_snode_last_use, and snodes_allocfree dictionaries
    to reflect the new memory state after swapping candidate with group.

    Args:
        candidate: Node that was moved
        gns: Group nodes
        group_tail: Last node of group
        candidate_delta_mem: Net memory change from candidate (alloc - free)
        candidate_allocfree: Candidate's allocation/free info
        group_n_to_bufs_after_swap_dealloc_by_candidate: Buffers whose deallocation moves to candidate
        post_alloc_update: Cached post-allocation memory values
        curr_memory: Current memory state dict (mutated)
        buf_to_snode_last_use: Buffer to last-use node mapping (mutated)
        snodes_allocfree: Node allocation/free info dict (mutated)
    """
    if not group_n_to_bufs_after_swap_dealloc_by_candidate:
        for gn in gns:
            cm = curr_memory[gn]
            curr_memory[gn] = (
                cm[0] - candidate_delta_mem,
                cm[1] - candidate_delta_mem,
            )
        _candidate_post_alloc_mem = (
            curr_memory[group_tail][1] + candidate_allocfree.size_alloc
        )
        _candidate_post_free_mem = (
            _candidate_post_alloc_mem - candidate_allocfree.size_free
        )
        curr_memory[candidate] = (
            _candidate_post_alloc_mem,
            _candidate_post_free_mem,
        )
        return

    # Candidate becomes last use of some bufs
    for bufs in group_n_to_bufs_after_swap_dealloc_by_candidate.values():
        for buf in bufs:
            buf_to_snode_last_use[buf] = candidate

    size_free_to_move_to_candidate_sum: int = 0
    for n in gns:
        _gn_post_alloc_mem: int = post_alloc_update[n]
        size_free_to_move_to_candidate: int = sum(
            buf.mpi_buffer.size_free
            for buf in group_n_to_bufs_after_swap_dealloc_by_candidate[n]
        )
        size_free_to_move_to_candidate_sum += size_free_to_move_to_candidate
        # group node does not deallocate this after swap
        snodes_allocfree[n].size_free -= size_free_to_move_to_candidate
        gn_post_free_mem: int = _gn_post_alloc_mem - snodes_allocfree[n].size_free
        curr_memory[n] = (_gn_post_alloc_mem, gn_post_free_mem)
    _candidate_post_alloc_mem = post_alloc_update[candidate]
    snodes_allocfree[candidate].size_free += size_free_to_move_to_candidate_sum
    candidate_post_free_mem = (
        _candidate_post_alloc_mem - snodes_allocfree[candidate].size_free
    )
    curr_memory[candidate] = (
        _candidate_post_alloc_mem,
        candidate_post_free_mem,
    )


def _find_buffers_with_changed_last_use(
    candidate: BaseSchedulerNode,
    gns: list[BaseSchedulerNode],
    buf_to_snode_last_use: dict,
    candidate_buffer_map: dict[BaseSchedulerNode, OrderedSet],
) -> dict[BaseSchedulerNode, list[FreeableInputBuffer | Any]]:
    """
    Find buffers whose last use will change after swapping candidate with group.

    When we swap [candidate [group]] to [[group] candidate], some buffers that
    were last used by a group node will now be last used by candidate instead.
    This affects memory deallocation timing.

    Args:
        candidate: The node being moved
        gns: Group nodes being swapped with candidate
        buf_to_snode_last_use: Mapping of buffers to their current last-use nodes
        candidate_buffer_map: Pre-computed map of node -> buffers using that node

    Returns:
        Dict mapping group nodes to buffers that will change their last-use node
    """
    group_n_to_bufs_after_swap_dealloc_by_candidate: dict[
        BaseSchedulerNode, list[FreeableInputBuffer | Any]
    ] = defaultdict(list)

    # Optimization: only check buffers where candidate is a successor
    # Reduces from O(all_buffers) to O(buffers_per_candidate)
    candidate_bufs = candidate_buffer_map.get(candidate, OrderedSet())
    gns_set = OrderedSet(gns)  # O(1) membership testing

    for buf in candidate_bufs:
        snode_last_use = buf_to_snode_last_use[buf]
        if snode_last_use in gns_set:
            group_n_to_bufs_after_swap_dealloc_by_candidate[snode_last_use].append(buf)

    return group_n_to_bufs_after_swap_dealloc_by_candidate


def _is_node_groupable_for_reorder(
    candidate: BaseSchedulerNode,
) -> tuple[bool, str | None]:
    """
    Check if a candidate node can be grouped with collective during reordering.

    This pass processes collectives left to right, so we avoid grouping with
    already-processed collectives based on configuration.

    Args:
        candidate: Node to check for groupability

    Returns:
        Tuple of (is_groupable, reason_if_not_groupable)
    """
    # This pass processes collectives left to right,
    # Do not group with processed collectives.
    # Leaving config for experimentation in 2D
    if not config_comms.reorder_iterative_group_with_collectives:
        if contains_async_collective(candidate):
            return (
                False,
                f"candidate contains_collective {candidate.get_name()}",
            )
    if not config_comms.reorder_iterative_use_runtime_estimations:
        if contains_gemm_like(candidate):
            return False, "contains_gemm_like"
    return True, None


def _format_and_log_reordering_stats(
    stats: dict[BaseSchedulerNode, ReorderInfo],
    head: BaseSchedulerNode,
    next_dict: dict[BaseSchedulerNode, BaseSchedulerNode | None],
    original_snodes_num: int,
    peak_memory: int,
    name_to_freeable_input_buf: dict,
    graph_outputs: OrderedSet[str],
) -> list[BaseSchedulerNode]:
    """
    Format reordering statistics, log them, and return final node list.

    Computes improvement metrics, creates a formatted table (using tabulate if
    available), validates the reordered node count, recalculates peak memory,
    and logs all information.

    Args:
        stats: Per-node reordering statistics
        head: Head of the reordered linked list
        next_dict: Linked list next pointers
        original_snodes_num: Original number of nodes (for validation)
        peak_memory: Initial peak memory before reordering
        name_to_freeable_input_buf: Buffer memory tracking info
        graph_outputs: Graph output names

    Returns:
        Final reordered list of scheduler nodes
    """
    node_stats = stats
    improvement = {snode: node_stats[snode].improvement for snode in node_stats}
    total_improvement = sum([improvement[snode] for snode in improvement])
    total_moves = sum([node_stats[snode].moves for snode in node_stats])

    reorder_log_str = (
        f"reorder_communication_preserving_peak_memory improved overlap by {total_improvement} ns"
        f" after {total_moves} reorders.\n"
    )
    headers = [
        "Collective node",
        "comm_time(us)",
        "comp_time(us)",
        "initial exposed(us)",
        "final exposed(us)",
        "improvement(us)",
        "limiting factor",
        "moves",
        "grouped",
        "grouped_info",
        "overlap_info",
    ]
    rows = [
        [
            node_summary(snode),
            node_info.comm_time / 1e3,
            node_info.comp_time / 1e3,
            node_info.initial_exposed / 1e3,
            node_info.final_exposed / 1e3,
            node_info.improvement / 1e3,
            node_info.limiting_factor,
            node_info.moves,
            node_info.grouped,
            node_info.grouped_info,
            node_info.overlap_info,
        ]
        for snode, node_info in node_stats.items()
    ]
    if importlib.util.find_spec("tabulate"):
        from tabulate import tabulate

        reorder_log_str += tabulate(
            rows,
            headers=headers,
        )
    else:
        reorder_log_str += (
            "Please `pip install tabulate` to nicely render overlap stats.\n"
        )
        reorder_log_str += str(headers) + "\n"
        reorder_log_str += "\n".join(map(str, rows))

    new_snodes = _group_nodes_from_linked_list(head, None, next_dict)
    assert len(new_snodes) == original_snodes_num
    new_peak_memory, _, _, _ = estimate_peak_memory_allocfree(
        new_snodes, name_to_freeable_input_buf, graph_outputs
    )
    reorder_log_str += f"\n peak_memory_before:{peak_memory}"
    reorder_log_str += f"\n peak_memory_after:{new_peak_memory}"

    overlap_log.info(reorder_log_str)
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "reorder_communication_preserving_peak_memory",
            "encoding": "string",
        },
        payload_fn=lambda: reorder_log_str,
    )

    return new_snodes


def _reorder_communication_preserving_peak_memory_internal(
    snodes: list[BaseSchedulerNode],
) -> tuple[list[BaseSchedulerNode], dict[BaseSchedulerNode, ReorderInfo]]:
    """
    Internal testing helper that also returns debug info.
    Returns:
        - reordered snodes list
        - dict {snode: ReorderInfo}
    """
    has_collectives = False
    for snode in snodes:
        if contains_collective(snode):
            has_collectives = True
            break
    if not has_collectives:
        return snodes, {}

    original_snodes_num = len(snodes)
    # heuristic to avoid degenerating to quadratic time
    graph_inputs: OrderedSet[str] = OrderedSet(V.graph.graph_inputs.keys())
    graph_outputs: OrderedSet[str] = OrderedSet(V.graph.get_output_names())
    (
        peak_memory,
        _curr_memory,
        snodes_allocfree,
        buf_to_snode_last_use,
        name_to_freeable_input_buf,
        candidate_buffer_map,
    ) = _initialize_memory_tracking(snodes, graph_inputs, graph_outputs)

    runtimes: dict[BaseSchedulerNode, float] = {
        snode: estimate_op_runtime(snode) * _op_runtime_estimate_mult(snode)
        for snode in snodes
    }

    # Pre-compute output and dependency sets for O(1) lookup instead of O(N) creation per iteration
    node_output_sets: dict[BaseSchedulerNode, frozenset[str]] = {
        snode: frozenset(o.get_name() for o in snode.get_outputs()) for snode in snodes
    }
    node_dep_sets: dict[BaseSchedulerNode, frozenset[str]] = {
        snode: frozenset(
            d.name for d in snode.unmet_dependencies if not _is_fake_dep(d)
        )
        for snode in snodes
    }

    # debug stats
    stats: dict[BaseSchedulerNode, ReorderInfo] = {}

    total_moves = 0

    _prev, _next, _head = _initialize_double_linked_list(snodes)

    debug_num_collectives_to_reorder: int | None = (
        config_comms.reorder_iterative_debug_limit_to_reorder
    )

    num_processed_collectives: int = 0
    curr: BaseSchedulerNode | None = _head
    debug_iterative_memory_recompute = (
        config_comms.reorder_iterative_debug_memory_recompute
    )
    iterative_recompute_error = False

    while curr is not None and _next[curr] is not None:
        _next_curr = _next[curr]
        if iterative_recompute_error:
            break

        if not contains_async_collective(curr):
            curr = _next_curr
            continue

        if debug_num_collectives_to_reorder is not None and (
            num_processed_collectives >= debug_num_collectives_to_reorder
        ):
            break
        num_processed_collectives += 1

        info = stats[curr] = ReorderInfo()
        comm_time, comp_time, overlap_info = _coll_exposed_communication_time(
            curr, _next, runtimes, node_output_sets, node_dep_sets
        )
        info.comm_time = comm_time
        info.comp_time = comp_time
        info.initial_exposed = info.final_exposed = comm_time - comp_time
        info.overlap_info = overlap_info

        candidate = _prev[curr]
        group_head = curr
        group_tail = curr
        group_waits = {}
        group_runtime = 0.0
        group_peak_memory = _curr_memory[curr][0]  # post_alloc memory

        # Track group dependencies incrementally - initialize from pre-computed sets
        group_unmet_deps_names = OrderedSet(node_dep_sets[curr])
        group_output_names = OrderedSet(node_output_sets[curr])

        while candidate is not None:
            if config_comms.reorder_iterative_use_runtime_estimations and (
                info.final_exposed
                < -config_comms.reorder_iterative_extra_comm_comp_overlap
                * info.comm_time
            ):
                info.limiting_factor = "unexposed by runtime estimations"
                break

            if (
                not config_comms.reorder_iterative_unsafe_collectives_reorder
                and contains_collective(candidate)
            ):
                info.limiting_factor = "collective ordering"
                break

            # Early exit: if group has no unmet dependencies, candidate can't have data dependency
            data_deps_names = group_unmet_deps_names - group_output_names
            if not data_deps_names:
                data_dep = False
            else:
                # Calculate effective dependencies (not satisfied within group)
                # Use pre-computed set for O(1) lookup
                candidate_out_names = node_output_sets[candidate]
                data_dep = bool(candidate_out_names & data_deps_names)

            if data_dep:
                is_groupable_result, grouping_reason = _is_node_groupable_for_reorder(
                    candidate
                )
                if is_groupable_result:
                    group_head = candidate

                    # Update incremental dependency tracking using pre-computed sets
                    group_unmet_deps_names.update(node_dep_sets[candidate])
                    group_output_names.update(node_output_sets[candidate])

                    if config_comms.reorder_iterative_use_runtime_estimations:
                        if contains_wait(candidate):
                            comm_time, comp_time, _ = _wait_exposed_communication_time(
                                candidate,
                                _head,
                                _prev,
                                runtimes,
                                node_output_sets,
                                node_dep_sets,
                            )
                            group_waits[candidate] = comm_time, comp_time
                        if not contains_async_collective(candidate):
                            group_runtime += runtimes[candidate]

                    group_peak_memory = max(
                        group_peak_memory, _curr_memory[candidate][0]
                    )
                    info.grouped += 1
                    candidate = _prev[candidate]
                    continue
                else:
                    msg = (
                        f"data dependency detected"
                        f"\n candidate:{candidate.get_name()}(outs:{[o.get_name() for o in candidate.get_outputs()]})"
                        f"\n non_group_reason:{grouping_reason}"
                    )
                    info.limiting_factor = msg
                    break

            if config_comms.reorder_iterative_use_runtime_estimations:
                # Check if candidate has sync runtime
                if not contains_async_collective(candidate):
                    c_runtime = runtimes[candidate]

                    if c_runtime > 0 and len(group_waits) > 0:
                        # pyrefly: ignore[no-matching-overload]
                        exposed_before = max(0, info.comm_time - info.comp_time)
                        # pyrefly: ignore[no-matching-overload]
                        exposed_after = max(
                            0, info.comm_time - info.comp_time - c_runtime
                        )
                        exposed_delta = exposed_after - exposed_before
                        for gw_comm_time, gw_comp_time in group_waits.values():
                            # pyrefly: ignore [no-matching-overload]
                            gw_exposed_before = max(0, gw_comm_time - gw_comp_time)
                            # pyrefly: ignore [no-matching-overload]
                            gw_exposed_after = max(
                                0, gw_comm_time - gw_comp_time + c_runtime
                            )

                            exposed_delta += gw_exposed_after - gw_exposed_before

                        if exposed_delta > 0:
                            info.limiting_factor = (
                                f"candidate has compute {c_runtime},"
                                f" group contains waits, total_exposed_delta {exposed_delta}"
                            )
                            break
                        else:
                            # Update all group_colls comm_time, comp_time
                            for gw, (
                                gw_comm_time,
                                gw_comp_time,
                            ) in group_waits.items():
                                group_waits[gw] = (
                                    gw_comm_time,
                                    gw_comp_time - c_runtime,
                                )
                else:
                    # Candidate is async_collective

                    # Unsafe collectives reordering
                    # Cj -> [...group_runtime..., Ci] -> Wj
                    # Checking that we are not increasing exposed time of Cj
                    if group_runtime > 0:
                        comm_time, comp_time, _ = _coll_exposed_communication_time(
                            candidate, _next, runtimes, node_output_sets, node_dep_sets
                        )
                        # pyrefly: ignore[no-matching-overload]
                        exposed_before = max(0, comm_time - comp_time)
                        # pyrefly: ignore[no-matching-overload]
                        exposed_after = max(0, comm_time - comp_time + group_runtime)
                        exposed_delta = exposed_after - exposed_before
                        if exposed_delta > 0:
                            info.limiting_factor = (
                                f"candidate {candidate.get_name()} is collective,"
                                f" group_runtime:{group_runtime},"
                                f" exposed_delta:{exposed_delta} c_comm_time:{comm_time} c_comp_time:{comp_time}"
                            )
                            break

            # Create group nodes list once for swap operations
            gns: list[BaseSchedulerNode] = _group_nodes_from_linked_list(
                group_head, group_tail, _next
            )

            candidate_allocfree: SNodeMemory = snodes_allocfree[candidate]
            candidate_delta_mem: int = (
                candidate_allocfree.size_alloc - candidate_allocfree.size_free
            )
            # candidate and one of group nodes are successors of the same buffer
            # and last use of the buffer happen in group nodes.
            # This last use deallocates it.
            # If we swap [candidate [group]] to [[group] candidate],
            # candidate becomes the last use
            # and deallocated this buffer instead of group node.
            # we need to update size_free accordingly to group_node and candidate,
            # and recalculate post_alloc, post_free for them.
            #
            # Buf that changes its last use snode,
            # after swap will be deallocated only by candidate,
            # while before it was deallocated by group node.
            group_n_to_bufs_after_swap_dealloc_by_candidate = (
                _find_buffers_with_changed_last_use(
                    candidate, gns, buf_to_snode_last_use, candidate_buffer_map
                )
            )

            potential_peak, _post_alloc_update = (
                _calculate_potential_peak_memory_reorder(
                    candidate,
                    gns,
                    group_tail,
                    group_peak_memory,
                    candidate_delta_mem,
                    candidate_allocfree,
                    group_n_to_bufs_after_swap_dealloc_by_candidate,
                    _curr_memory,
                )
            )

            if (
                potential_peak - peak_memory
                > peak_memory * config_comms.reorder_iterative_peak_memory_budget
            ):
                info.limiting_factor = (
                    f"peak memory new:{potential_peak} vs base:{peak_memory}"
                )
                break
            info.moves += 1
            total_moves += 1

            _head = _perform_double_linked_list_swap(
                candidate, group_head, group_tail, _prev, _next, _head
            )

            comm_time, comp_time, overlap_info = _coll_exposed_communication_time(
                curr, _next, runtimes, node_output_sets, node_dep_sets
            )
            info.comm_time = comm_time
            info.comp_time = comp_time
            info.overlap_info = overlap_info
            info.final_exposed = comm_time - comp_time

            _update_memory_tracking_after_swap_reorder(
                candidate,
                gns,
                group_tail,
                candidate_delta_mem,
                candidate_allocfree,
                group_n_to_bufs_after_swap_dealloc_by_candidate,
                _post_alloc_update,
                _curr_memory,
                buf_to_snode_last_use,
                snodes_allocfree,
            )

            if debug_iterative_memory_recompute:
                # Compare iteratively recomputed memory data
                # with full run of estimate_peak_memory

                from .comms_debug import _debug_iterative_memory_recompute

                iterative_recompute_error = _debug_iterative_memory_recompute(
                    candidate,
                    gns,
                    _group_names(gns),
                    _group_nodes_from_linked_list(_head, None, _next),
                    name_to_freeable_input_buf,
                    graph_outputs,
                    peak_memory,
                    _curr_memory,
                    snodes_allocfree,
                    "reorder_communication_preserving_peak_memory",
                    group_n_to_bufs_after_swap_dealloc_by_candidate,
                )
                if iterative_recompute_error:
                    break
            candidate = _prev[group_head]
        curr = _next_curr

    if not config_comms.reorder_sink_verbose_logging:
        new_snodes = _group_nodes_from_linked_list(_head, None, _next)
        return new_snodes, stats

    new_snodes = _format_and_log_reordering_stats(
        stats,
        _head,
        _next,
        original_snodes_num,
        peak_memory,
        name_to_freeable_input_buf,
        graph_outputs,
    )

    return new_snodes, stats


def _schedule_for_comm(
    snodes: list[BaseSchedulerNode],
    raise_comms: bool,
    sink_waits: bool,
    reorder_for_overlap: bool,
) -> list[BaseSchedulerNode]:
    """
    Schedule `snodes` for various comm optimization objectives.

    Args:
        snodes: the nodes to be scheduled.
        raise_comms: whether to greedily schedule collectives as early as possible
        sink_wait: whether to greedily schedule waits as late as possible
        reorder_compute_for_overlap: whether to reorder compute nodes to
            optimize for compute/communication overlapping.

    Returns:
        The new schedule order.

    Some notes on the synergy between different options:
        - `raise_comms` provides more overlapping oppurtunies for `reorder_compute_for_overlap`.
        - When both `raise_comms` and `sink_waits` is `True`, `raise_comms` is prioritized.
    """
    # We assign each node a tuple of scores (score_0, score_1, score_2),
    # decreasing in importance, with a lower value indicating a higher ranking:
    #
    # - score_0: the lowest comm_idx among the comm nodes that the node blocks.
    # If a node doesn't block any comm nodes, its score_0 is set to
    # sys.maxsize. This score ensures that comm nodes get scheduled as early as
    # possible.
    # - score_1: 1 if the node is a wait node, 0 otherwise. This score ensures
    # that wait nodes are deferred as late as possible.
    # - score_2: the index of the node in the original topological order. This
    # score provides stability in case of ties.
    #
    # When only raise_comms is True, only score_0 and score_2 are considered.
    # When only sink_waits is True, only score_1 and score_2 are considered.
    # When neither is True, the original order is yielded.
    buf_name_to_snode = {}
    name_to_fused_node = {}
    scores_0, scores_1, scores_2 = {}, {}, {}
    for idx, snode in enumerate(snodes):
        for buf_name in snode.get_buffer_names():
            buf_name_to_snode[buf_name] = snode

        for op_name in snode.get_operation_names():
            name_to_fused_node[op_name] = snode
        name_to_fused_node[snode.get_name()] = snode

        node_name = snode.get_name()
        scores_0[node_name] = sys.maxsize
        scores_1[node_name] = 0
        scores_2[node_name] = idx

    comm_idx = 0
    for snode in snodes:
        if raise_comms and contains_collective(snode):
            scores_0[snode.get_name()] = comm_idx
            for ancestor in snode.ancestors:
                anc_fused_name = name_to_fused_node[ancestor].get_name()
                scores_0[anc_fused_name] = min(scores_0[anc_fused_name], comm_idx)
            comm_idx += 1
        elif sink_waits and contains_wait(snode):
            scores_1[snode.get_name()] = 1

    class Runnable:
        def __init__(self, snode) -> None:
            self.snode = snode
            name = next(iter(snode.get_operation_names()))
            fused_name = name_to_fused_node[name].get_name()
            self.score = (
                scores_0[fused_name],
                scores_1[fused_name],
                scores_2[fused_name],
            )

        def __lt__(self, other):
            return self.score < other.score

    unmet_deps: dict[BaseSchedulerNode, OrderedSet[str]] = {
        snode: OrderedSet(dep.name for dep in snode.unmet_dependencies)
        for snode in snodes
    }

    ready: list[Runnable] = []
    buffer_users: dict[str, OrderedSet[BaseSchedulerNode]] = defaultdict(OrderedSet)
    snode_to_cost = {snode: estimate_op_runtime(snode) for snode in snodes}

    for snode, deps in unmet_deps.items():
        if len(deps) == 0:
            heapq.heappush(ready, Runnable(snode))
        for dep in deps:
            buffer_users[dep].add(snode)

    scheduled = []

    def schedule(snode):
        """
        Schedules `snode` and put all unblocked nodes onto the ready queue.
        """
        scheduled.append(snode)
        for buf_name in snode.get_buffer_names():
            for snode in buffer_users[buf_name]:
                unmet_deps[snode].remove(buf_name)
                if len(unmet_deps[snode]) == 0:
                    heapq.heappush(ready, Runnable(snode))

    def get_overlapping_candidate():
        """
        Return the next node in the ready queue that's neither a collective or
        a wait.
        """
        candidates = [
            x
            for x in ready
            if not contains_collective(x.snode) and not contains_wait(x.snode)
        ]
        if len(candidates) == 0:
            return None
        return min(candidates, key=lambda x: x.score)

    def schedule_collective_for_overlap(snode):
        """
        Schedules collective node `snode`, along with one or more compute nodes
        to overlap with it. The strategy is described in the comment of
        `reorder_compute_for_overlap`.
        """
        assert contains_collective(snode)
        schedule(snode)

        collective_cost = snode_to_cost[snode]
        while (
            collective_cost > 0
            and (candidate := get_overlapping_candidate()) is not None
        ):
            ready.remove(candidate)

            schedule(candidate.snode)

            collective_cost -= snode_to_cost[candidate.snode]
        heapq.heapify(ready)

    while ready:
        snode = heapq.heappop(ready).snode
        if reorder_for_overlap and contains_collective(snode):
            schedule_collective_for_overlap(snode)
        else:
            schedule(snode)

    for deps in unmet_deps.values():
        assert len(deps) == 0, (
            f"Detected unscheduled nodes. Nodes with unmet dependencies: {unmet_deps}"
        )
    return scheduled


def decide_global_ordering_of_comms(
    nodes: list[BaseSchedulerNode], name_to_buf, name_to_fused_node
) -> list[BaseSchedulerNode]:
    """
    Decide global ordering of comms, by just enforcing the ordering that's in the input graph
    (might not be the same ordering as the eager mode program).
    TODO: Come up with a better approach
    """
    if not torch.distributed.is_available():
        return nodes

    comm_nodes = [n for n in nodes if contains_collective(n)]

    for i in range(1, len(comm_nodes)):
        # Enforce ordering by making previous comm a `WeakDep` dependency of the next comm
        mutating_buf = next(iter(comm_nodes[i].get_buffer_names()))
        for buf in comm_nodes[i - 1].get_buffer_names():
            comm_nodes[i].add_fake_dep(
                WeakDep(buf, mutating_buf=mutating_buf, is_fake=True)
            )

    return nodes


@dataclass
class SinkWaitInfo:
    grouped: int = 0
    grouped_info: str = ""
    moves: int = 0
    moves_info: str = ""
    limiting_factor: str = "None"
    comm_time: float = -1.0
    comp_time: float = -1.0
    initial_exposed: float = -1.0
    final_exposed: float = -1.0
    overlap_info: str = "None"

    @property
    def improvement(self):
        return self.initial_exposed - self.final_exposed


def _is_node_groupable_for_sink_waits(
    candidate: BaseSchedulerNode,
) -> tuple[bool, str | None]:
    """
    Check if a candidate node can be grouped during sink_waits pass.

    Sink Waits traverses waits right to left, so we don't group with
    processed waits on the right or with async collectives.

    Args:
        candidate: Node to check for groupability

    Returns:
        Tuple of (is_groupable, reason_if_not_groupable)
    """
    # Sink Waits traverse Waits right to left,
    # => we do not group with processed Waits on the right.
    if contains_wait(candidate):
        return False, f"candidate contains wait {candidate.get_name()}"
    if contains_async_collective(candidate):
        return (
            False,
            f"candidate contains_async_collective {candidate.get_name()}",
        )

    if not config_comms.sink_iterative_use_runtime_estimations:
        # Heuristics pre-use_runtime_estimations:
        # TODO(ivankobzarev): Remove them after confirming,
        # that using runtime estimations always give better results.
        # We do not want to group with collectives to not reorder them forward.
        if contains_collective(candidate):
            return (
                False,
                f"candidate contains collective {candidate.get_name()}",
            )
        if contains_gemm_like(candidate):
            return (
                False,
                f"candidate contains gemm_like {candidate.get_name()}",
            )
    return True, None


def _update_memory_tracking_after_swap_sink_waits(
    candidate: BaseSchedulerNode,
    gns: list[BaseSchedulerNode],
    candidate_delta_mem: int,
    candidate_allocfree: SNodeMemory,
    group_n_to_bufs_after_swap_dealloc_instead_of_candidate: dict,
    post_alloc_update: dict[BaseSchedulerNode, int],
    size_free_delta_update: dict[BaseSchedulerNode, int],
    curr_memory: dict,
    snodes_allocfree: dict,
) -> None:
    """
    Update memory tracking structures after swap (sink_waits version).

    Updates curr_memory and snodes_allocfree dictionaries to reflect the new
    memory state after swapping candidate with group.

    Args:
        candidate: Node that was moved
        gns: Group nodes
        candidate_delta_mem: Net memory change from candidate (alloc - free)
        candidate_allocfree: Candidate's allocation/free info
        group_n_to_bufs_after_swap_dealloc_instead_of_candidate: Buffers whose deallocation moves from candidate to group
        post_alloc_update: Cached post-allocation memory values
        size_free_delta_update: Cached size-free delta values
        curr_memory: Current memory state dict (mutated)
        snodes_allocfree: Node allocation/free info dict (mutated)
    """
    group_head = gns[0]
    pre_group_mem = curr_memory[group_head][0] - snodes_allocfree[group_head].size_alloc
    if not group_n_to_bufs_after_swap_dealloc_instead_of_candidate:
        candidate_post_alloc = pre_group_mem + candidate_allocfree.size_alloc
        curr_memory[candidate] = (
            candidate_post_alloc,
            candidate_post_alloc - candidate_allocfree.size_free,
        )
        for gn in gns:
            cm = curr_memory[gn]
            curr_memory[gn] = (
                cm[0] + candidate_delta_mem,
                cm[1] + candidate_delta_mem,
            )
        return

    for n in [candidate, *gns]:
        post_alloc = post_alloc_update[n]
        snodes_allocfree[n].size_free += size_free_delta_update.get(n, 0)
        curr_memory[n] = (
            post_alloc,
            post_alloc - snodes_allocfree[n].size_free,
        )


def _calculate_potential_peak_memory_sink_waits(
    candidate: BaseSchedulerNode,
    gns: list[BaseSchedulerNode],
    group_head: BaseSchedulerNode,
    group_peak_memory: int,
    candidate_delta_mem: int,
    candidate_allocfree: SNodeMemory,
    group_n_to_bufs_after_swap_dealloc_instead_of_candidate: dict,
    curr_memory: dict,
    snodes_allocfree: dict,
) -> tuple[int, dict[BaseSchedulerNode, int], dict[BaseSchedulerNode, int]]:
    """
    Calculate potential peak memory after swapping candidate with group (sink_waits version).

    Computes new memory levels for all affected nodes and returns the potential
    peak memory along with cached post-allocation and size-free delta values.

    Args:
        candidate: Node being moved
        gns: Group nodes
        group_head: First node of group
        group_peak_memory: Current peak memory within the group
        candidate_delta_mem: Net memory change from candidate (alloc - free)
        candidate_allocfree: Candidate's allocation/free info
        group_n_to_bufs_after_swap_dealloc_instead_of_candidate: Buffers whose deallocation moves from candidate to group
        curr_memory: Current memory state dict
        snodes_allocfree: Allocation/free info for all nodes

    Returns:
        Tuple of (potential_peak_memory, post_alloc_update_dict, size_free_delta_update_dict)
    """
    pre_group_mem = curr_memory[group_head][0] - snodes_allocfree[group_head].size_alloc
    # Stash memory tracing updates to not recompute them after swap
    _post_alloc_update: dict[BaseSchedulerNode, int] = {}
    _size_free_delta_update: dict[BaseSchedulerNode, int] = {}

    potential_peak = 0
    if not group_n_to_bufs_after_swap_dealloc_instead_of_candidate:
        # Not accounting for buffers liveliness change
        potential_peak = max(
            group_peak_memory + candidate_delta_mem,
            pre_group_mem + candidate_allocfree.size_alloc,
        )
        return potential_peak, _post_alloc_update, _size_free_delta_update

    candidate_post_alloc = pre_group_mem + candidate_allocfree.size_alloc
    _post_alloc_update[candidate] = candidate_post_alloc
    potential_peak = candidate_post_alloc
    candidate_size_free_to_move = sum(
        buf.mpi_buffer.size_free  # type: ignore[attr-defined]
        for buf in itertools.chain.from_iterable(
            group_n_to_bufs_after_swap_dealloc_instead_of_candidate.values()
        )
    )
    _size_free_delta_update[candidate] = -candidate_size_free_to_move
    delta_mem = candidate_delta_mem + candidate_size_free_to_move
    for gn in gns:
        gn_post_alloc = curr_memory[gn][0] + delta_mem
        _post_alloc_update[gn] = gn_post_alloc
        potential_peak = max(potential_peak, gn_post_alloc)
        gn_size_free_to_add = 0
        if gn in group_n_to_bufs_after_swap_dealloc_instead_of_candidate:
            bufs = group_n_to_bufs_after_swap_dealloc_instead_of_candidate[gn]
            for buf in bufs:
                gn_size_free_to_add += buf.mpi_buffer.size_free
            _size_free_delta_update[gn] = gn_size_free_to_add
        delta_mem -= gn_size_free_to_add
    return potential_peak, _post_alloc_update, _size_free_delta_update


def _perform_double_linked_list_swap_sink_waits(
    candidate: BaseSchedulerNode,
    group_head: BaseSchedulerNode,
    group_tail: BaseSchedulerNode,
    prev_dict: dict[BaseSchedulerNode, BaseSchedulerNode | None],
    next_dict: dict[BaseSchedulerNode, BaseSchedulerNode | None],
    head: BaseSchedulerNode,
) -> BaseSchedulerNode:
    """
    Swap positions of candidate and group in doubly-linked list (sink_waits version).

    Transforms (moves candidate to the left):
    group_head_prev -> group_head...group_tail -> candidate -> candidate_next
    Into:
    group_head_prev -> candidate -> group_head...group_tail -> candidate_next

    Args:
        candidate: Node to swap with group
        group_head: First node of group
        group_tail: Last node of group
        prev_dict: Dictionary mapping nodes to their previous nodes
        next_dict: Dictionary mapping nodes to their next nodes
        head: Current head of the linked list

    Returns:
        New head of the linked list (may change if group_head was the head)
    """
    # 0: Update group_head's previous node
    group_head_prev = prev_dict[group_head]
    if group_head_prev:
        next_dict[group_head_prev] = candidate
    prev_dict[candidate] = group_head_prev

    # 2: Update candidate's next node
    candidate_next = next_dict[candidate]
    if candidate_next:
        prev_dict[candidate_next] = group_tail
    next_dict[group_tail] = candidate_next

    # 1: Link candidate to group_head
    prev_dict[group_head] = candidate
    next_dict[candidate] = group_head

    # Update head if group_head was the head
    if group_head == head:
        return candidate
    return head


def _format_and_log_sink_waits_stats(
    stats: dict[BaseSchedulerNode, SinkWaitInfo],
    head: BaseSchedulerNode,
    next_dict: dict[BaseSchedulerNode, BaseSchedulerNode | None],
    original_snodes_num: int,
    peak_memory: int,
    name_to_freeable_input_buf: dict,
    graph_outputs: OrderedSet[str],
) -> list[BaseSchedulerNode]:
    """
    Format sink_waits statistics, log them, and return final node list.

    Computes improvement metrics, creates a formatted table (using tabulate if
    available), validates the reordered node count, recalculates peak memory,
    and logs all information.

    Args:
        stats: Per-node sink_waits statistics
        head: Head of the reordered linked list
        next_dict: Linked list next pointers
        original_snodes_num: Original number of nodes (for validation)
        peak_memory: Initial peak memory before reordering
        name_to_freeable_input_buf: Buffer memory tracking info
        graph_outputs: Graph output names

    Returns:
        Final reordered list of scheduler nodes
    """
    headers = [
        "Wait node",
        "comm_time(us)",
        "comp_time(us)",
        "initial exposed(us)",
        "final exposed(us)",
        "improvement(us)",
        "limiting factor",
        "grouped",
        "grouped_info",
        "moves",
        "moves_info",
        "overlap_info",
    ]
    rows = [
        [
            node_summary(snode),
            info.comm_time / 1e3,
            info.comp_time / 1e3,
            info.initial_exposed / 1e3,
            info.final_exposed / 1e3,
            info.improvement / 1e3,
            info.limiting_factor,
            info.grouped,
            info.grouped_info,
            info.moves,
            info.moves_info,
            info.overlap_info,
        ]
        for snode, info in stats.items()
    ]
    log_str = ""
    if importlib.util.find_spec("tabulate"):
        from tabulate import tabulate

        log_str += tabulate(
            rows,
            headers=headers,
        )
    else:
        log_str += "Please `pip install tabulate` to nicely render overlap stats.\n"
        log_str += str(headers) + "\n"
        log_str += "\n".join(map(str, rows))
    overlap_log.info(log_str)
    new_snodes = _group_nodes_from_linked_list(head, None, next_dict)
    assert len(new_snodes) == original_snodes_num
    new_peak_memory, _, _, _ = estimate_peak_memory_allocfree(
        new_snodes, name_to_freeable_input_buf, graph_outputs
    )
    log_str += f"\n sink_waits_iterative peak_memory_before:{peak_memory}"
    log_str += f"\n sink_waits_iterative peak_memory_after:{new_peak_memory}"
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "sink_waits_iterative_info",
            "encoding": "string",
        },
        payload_fn=lambda: log_str,
    )
    return new_snodes


def _find_buffers_with_changed_last_use_sink_waits(
    candidate: BaseSchedulerNode,
    gns: list[BaseSchedulerNode],
    buf_to_snode_last_use: dict,
    candidate_buffer_map: dict[BaseSchedulerNode, OrderedSet],
) -> dict[BaseSchedulerNode, list[FreeableInputBuffer | Any]]:
    """
    Find buffers whose last use will change after swapping in sink_waits pass.

    When we swap [group] candidate to candidate [group], some buffers that
    were last used by candidate will now be last used by a group node instead.
    This is the opposite direction from the reorder version.

    Args:
        candidate: The node being moved (currently last use)
        gns: Group nodes being swapped with candidate
        buf_to_snode_last_use: Mapping of buffers to their current last-use nodes
        candidate_buffer_map: Pre-computed map of node -> buffers using that node

    Returns:
        Dict mapping group nodes to buffers that will change their last-use node
    """
    group_n_to_bufs_after_swap_dealloc_instead_of_candidate: dict[
        BaseSchedulerNode, list[FreeableInputBuffer | Any]
    ] = defaultdict(list)

    # Optimization: only check buffers where candidate is a successor
    # Reduces from O(all_buffers) to O(buffers_per_candidate)
    candidate_bufs = candidate_buffer_map.get(candidate, OrderedSet())

    for buf in candidate_bufs:
        snode_last_use = buf_to_snode_last_use[buf]
        if snode_last_use != candidate:  # noqa: E711
            continue

        # candidate is last use of buf
        # Find last group node in successors (maintains order)
        succ_nodes = buf.mpi_buffer.succ_nodes
        last_succ_gn = None
        for gn in gns:
            if gn in succ_nodes:
                last_succ_gn = gn

        if last_succ_gn is None:
            continue

        # gn has successors of buf that after potential swap will become
        # last use of buf and start deallocating buf instead of candidate
        group_n_to_bufs_after_swap_dealloc_instead_of_candidate[last_succ_gn].append(
            buf
        )

    return group_n_to_bufs_after_swap_dealloc_instead_of_candidate


def _sink_waits_iterative_internal(
    snodes: list[BaseSchedulerNode],
) -> tuple[list[BaseSchedulerNode], dict[BaseSchedulerNode, SinkWaitInfo]]:
    original_snodes_num = len(snodes)
    if original_snodes_num == 0:
        return snodes, {}
    graph_inputs: OrderedSet[str] = OrderedSet(V.graph.graph_inputs.keys())
    graph_outputs: OrderedSet[str] = OrderedSet(V.graph.get_output_names())
    (
        peak_memory,
        _curr_memory,
        snodes_allocfree,
        buf_to_snode_last_use,
        name_to_freeable_input_buf,
        candidate_buffer_map,
    ) = _initialize_memory_tracking(snodes, graph_inputs, graph_outputs)

    _prev, _next, _head = _initialize_double_linked_list(snodes)

    stats: dict[BaseSchedulerNode, SinkWaitInfo] = {}

    runtimes: dict[BaseSchedulerNode, float] = {
        snode: estimate_op_runtime(snode) * _op_runtime_estimate_mult(snode)
        for snode in snodes
    }

    # Pre-compute output and dependency sets for O(1) lookup instead of O(N) creation per iteration
    node_output_sets: dict[BaseSchedulerNode, frozenset[str]] = {
        snode: frozenset(o.get_name() for o in snode.get_outputs()) for snode in snodes
    }
    node_dep_sets: dict[BaseSchedulerNode, frozenset[str]] = {
        snode: frozenset(
            d.name for d in snode.unmet_dependencies if not _is_fake_dep(d)
        )
        for snode in snodes
    }

    curr: BaseSchedulerNode | None = snodes[-1]

    processed_waits = OrderedSet()  # type: ignore[var-annotated]
    debug_iterative_memory_recompute = (
        config_comms.reorder_iterative_debug_memory_recompute
    )
    debug_num_sink_waits_to_reorder: int | None = (
        config_comms.sink_waits_iterative_debug_limit_to_sink
    )

    iterative_recompute_error = False
    while curr is not None and _prev[curr] is not None:
        _prev_curr = _prev[curr]
        if iterative_recompute_error:
            break
        if (
            debug_num_sink_waits_to_reorder is not None
            and len(processed_waits) >= debug_num_sink_waits_to_reorder
        ):
            break

        if not (contains_wait(curr) and curr not in processed_waits):
            curr = _prev_curr
            continue

        processed_waits.add(curr)
        info = stats[curr] = SinkWaitInfo()
        comm_time, comp_time, overlap_info = _wait_exposed_communication_time(
            curr, _head, _prev, runtimes, node_output_sets, node_dep_sets
        )
        info.initial_exposed = info.final_exposed = comm_time - comp_time
        info.comm_time = comm_time
        info.comp_time = comp_time
        info.overlap_info = overlap_info

        candidate = _next[curr]
        group_head = curr
        group_tail = curr
        group_colls = {}
        group_runtime = 0.0
        group_peak_memory = _curr_memory[curr][0]

        # Track group outputs and check collective status incrementally - initialize from pre-computed set
        group_output_names = OrderedSet(node_output_sets[curr])
        group_contains_collective = contains_collective(curr)

        while candidate is not None:
            if config_comms.sink_iterative_use_runtime_estimations and (
                info.final_exposed
                < -config_comms.sink_iterative_extra_comm_comp_overlap * info.comm_time
            ):
                info.limiting_factor = "unexposed by runtime estimations"
                break

            # Early exit: if group has no outputs, candidate can't depend on it
            if not group_output_names:
                data_dep = False
            else:
                # Calculate candidate dependencies using pre-computed set
                candidate_dep_names = node_dep_sets[candidate]
                data_dep = bool(candidate_dep_names & group_output_names)

            # Conservative sink wait, limiting by space before next collective.
            # The global strategy is that bucketing should create space.
            # For 2D we can experiment with allowing to sink Wait beyond non current group collective.

            if not config_comms.sink_waits_iterative_swap_with_collectives:
                if contains_async_collective(candidate):
                    info.limiting_factor = (
                        f"candidate contains_async_collective {candidate.get_name()}"
                    )
                    break

            # 1. If we have data_dep - we can not swap => trying to group
            # 2. If swap candidate and current node both contain collectives => trying to group
            both_contain_comms = group_contains_collective and contains_collective(
                candidate
            )
            if data_dep or both_contain_comms:
                _is_groupable, groupable_reason = _is_node_groupable_for_sink_waits(
                    candidate
                )
                if _is_groupable:
                    group_tail = candidate

                    # Update incremental tracking using pre-computed set
                    group_output_names.update(node_output_sets[candidate])
                    group_contains_collective = (
                        group_contains_collective or contains_collective(candidate)
                    )

                    if (
                        config_comms.sink_iterative_use_runtime_estimations
                        and contains_collective(candidate)
                    ):
                        comm_time, comp_time, _ = _coll_exposed_communication_time(
                            candidate, _next, runtimes, node_output_sets, node_dep_sets
                        )
                        group_colls[candidate] = (comm_time, comp_time)
                        if not contains_async_collective(candidate):
                            group_runtime += runtimes[candidate]

                    group_peak_memory = max(
                        group_peak_memory, _curr_memory[candidate][0]
                    )
                    info.grouped += 1
                    candidate = _next[candidate]
                    continue
                elif not data_dep:
                    if (
                        not config_comms.sink_waits_iterative_unsafe_collectives_reorder
                        and both_contain_comms
                    ):
                        info.limiting_factor = (
                            f"collective ordering"
                            f"\n with candidate:{candidate.get_name()}"
                        )
                        break
                else:
                    info.limiting_factor = (
                        f"data dependency detected"
                        f"\n candidate:{candidate.get_name()}"
                        f"\n non_group_reason:{groupable_reason}"
                    )
                    break

            if config_comms.sink_iterative_use_runtime_estimations:
                if is_wait(candidate.node):
                    # Corresponding collective is before the group,
                    # Swap can increase exposed time of corresponding collective
                    comm_time, comp_time, _ = _wait_exposed_communication_time(
                        candidate,
                        _head,
                        _prev,
                        runtimes,
                        node_output_sets,
                        node_dep_sets,
                    )
                    # pyrefly: ignore[no-matching-overload]
                    exposed_before = max(0, comm_time - comp_time)
                    # pyrefly: ignore[no-matching-overload]
                    exposed_after = max(0, comm_time - comp_time + group_runtime)
                    # We do not know how much we can sink more after this swap,
                    # Just comparing advantage at the moment for now.
                    if exposed_after > exposed_before:
                        info.limiting_factor = (
                            "candidate is wait,"
                            f" exposed_before:{exposed_before} vs exposed_after:{exposed_after}"
                        )
                        break

                # Check if candidate has sync runtime
                if not contains_async_collective(candidate):
                    # If candidate has sync runtime,
                    # Waits of gorup_colls are on the right from group.
                    # Swap can increase their exposed time.
                    c_runtime = runtimes[candidate]

                    if c_runtime > 0 and len(group_colls) > 0:
                        # Advantage for current Wait to do the Swap
                        # pyrefly: ignore[no-matching-overload]
                        exposed_delta = max(
                            0,
                            info.comm_time - info.comp_time,
                        )
                        # pyrefly: ignore[no-matching-overload]
                        -max(0, info.comm_time - info.comp_time - c_runtime)
                        for gc_comm_time, gc_comp_time in group_colls.values():
                            # pyrefly: ignore [no-matching-overload]
                            exposed_delta += max(0, gc_comm_time - gc_comp_time) - max(
                                0, gc_comm_time - gc_comp_time + c_runtime
                            )
                        if exposed_delta > 0:
                            info.limiting_factor = (
                                f"candidate has compute {c_runtime}, group contains collectives,"
                                f" total_exposed_delta {exposed_delta}"
                            )
                            break
                        else:
                            # Update all group_colls comm_time, comp_time
                            for gc, (
                                gc_comm_time,
                                gc_comp_time,
                            ) in group_colls.items():
                                group_colls[gc] = (
                                    gc_comm_time,
                                    gc_comp_time - c_runtime,
                                )

            # Create group nodes list once for swap operations
            gns: list[BaseSchedulerNode] = _group_nodes_from_linked_list(
                group_head, group_tail, _next
            )

            candidate_allocfree: SNodeMemory = snodes_allocfree[candidate]
            candidate_delta_mem = (
                candidate_allocfree.size_alloc - candidate_allocfree.size_free
            )
            # [group] candidate -> candidate [group]
            # Check for buffers with successors in group and candidate last successor
            #
            # Buf that  changes its last use snode,
            # It was deallocated by candidate,
            # but after swap it will be deallocated by group node.
            group_n_to_bufs_after_swap_dealloc_instead_of_candidate = (
                _find_buffers_with_changed_last_use_sink_waits(
                    candidate, gns, buf_to_snode_last_use, candidate_buffer_map
                )
            )

            potential_peak, _post_alloc_update, _size_free_delta_update = (
                _calculate_potential_peak_memory_sink_waits(
                    candidate,
                    gns,
                    group_head,
                    group_peak_memory,
                    candidate_delta_mem,
                    candidate_allocfree,
                    group_n_to_bufs_after_swap_dealloc_instead_of_candidate,
                    _curr_memory,
                    snodes_allocfree,
                )
            )
            if (
                potential_peak - peak_memory
                > peak_memory * config_comms.sink_iterative_peak_memory_budget
            ):
                info.limiting_factor = (
                    f"peak memory new:{potential_peak} vs base:{peak_memory}"
                )
                break

            info.moves += 1
            info.moves_info += f"+{candidate.get_name()}"

            _head = _perform_double_linked_list_swap_sink_waits(
                candidate, group_head, group_tail, _prev, _next, _head
            )

            comm_time, comp_time, overlap_info = _wait_exposed_communication_time(
                curr, _head, _prev, runtimes, node_output_sets, node_dep_sets
            )
            info.comm_time = comm_time
            info.comp_time = comp_time
            info.final_exposed = comm_time - comp_time
            info.overlap_info = overlap_info

            _update_memory_tracking_after_swap_sink_waits(
                candidate,
                gns,
                candidate_delta_mem,
                candidate_allocfree,
                group_n_to_bufs_after_swap_dealloc_instead_of_candidate,
                _post_alloc_update,
                _size_free_delta_update,
                _curr_memory,
                snodes_allocfree,
            )

            if debug_iterative_memory_recompute:
                from .comms_debug import _debug_iterative_memory_recompute

                iterative_recompute_error = _debug_iterative_memory_recompute(
                    candidate,
                    gns,
                    _group_names(gns),
                    _group_nodes_from_linked_list(_head, None, _next),
                    name_to_freeable_input_buf,
                    graph_outputs,
                    peak_memory,
                    _curr_memory,
                    snodes_allocfree,
                    "sink_waits_iterative",
                    group_n_to_bufs_after_swap_dealloc_instead_of_candidate,
                )
                if iterative_recompute_error:
                    break

            candidate = _next[group_tail]
        curr = _prev_curr

    if not config_comms.reorder_sink_verbose_logging:
        new_snodes = _group_nodes_from_linked_list(_head, None, _next)
        return new_snodes, stats

    new_snodes = _format_and_log_sink_waits_stats(
        stats,
        _head,
        _next,
        original_snodes_num,
        peak_memory,
        name_to_freeable_input_buf,
        graph_outputs,
    )

    return new_snodes, stats


def sink_waits_iterative(snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Similarly to reorder_communication_preserving_peak_memory this pass will try to iteratively
    push Wait nodes later, recomputing estimated peak memory before each swap,
    and preventing peak memory regressions.

    Pass will be applied to every Wait node. If there are immediate dependencies with next node,
    pass will try to group them together and on the next step to swap the group with next candidate.

    If _inductor.config_comms.sink_iterative_use_runtime_estimations is set True,
    pass will stop reordering of Wait once corresponding Collective is unexposed,
    based on runtime estimations.

    inductor.config_comms.sink_iterative_peak_memory_budget allows to tune how much pass
    can regress initial peak memory.
    E.g.:
    sink_iterative_peak_memory_budget == 0.0 - No regression of initial peak memory is allowed
    sink_iterative_peak_memory_budget == 0.2 - Pass can improve comm-compute overlap, sacrificing
    20% of initial peak memory value.

    inductor.config_comms.sink_iterative_extra_comm_comp_overlap config allows to more aggressively
    sink waits, stopping only when overlap_compute >= (1 + extra_comm_comp_overlap) * comm_time
    """
    return _sink_waits_iterative_internal(snodes)[0]


def estimate_op_runtime(snode: BaseSchedulerNode) -> float:
    """
    Returns estimated op runtime in milliseconds (ms)
    """
    if config.estimate_op_runtime == "default":
        runtime = snode.get_estimated_runtime()
    else:
        assert callable(config.estimate_op_runtime)
        runtime = config.estimate_op_runtime(snode)
    return runtime


def node_summary(snode):
    snodes = snode.get_nodes()
    if len(snodes) == 1:
        detail = ""
        if isinstance(snode.node, (ir.ExternKernelOut, ir._CollectiveKernel)):
            outs_str = f"outs:{[o.get_name() for o in snode.get_outputs()]}"
            ins_str = f"ins:{[d.name for d in snode.unmet_dependencies]}"
            detail = f" {snode.get_name()} ({snode.node.python_kernel_name})\n {outs_str}({ins_str})"
        layouts = [child.node.get_output_spec() for child in snode.get_nodes()]
        out_tensor_info = ",".join(
            [
                f" (size={layout.size}, stride={layout.stride})"
                if isinstance(layout, ir.Layout)
                else ""
                for layout in layouts
            ]
        )
        try:
            node_name = snode.node.maybe_get_name()
        except AttributeError:
            # TODO: node_summary was written without FusedSchedulerNode in mind, generally needs to be hardened
            node_name = ""
        return f"{snode.node.__class__.__name__}{detail}{out_tensor_info} ({node_name} ({snode.get_estimated_runtime():.0f} ns)"

    # Flatten the summaries for Fused/Foreach/Grouped nodes
    summaries = []
    for child_snode in snodes:
        summaries.append(node_summary(child_snode))
    return f"{snode.__class__.__name__}: {', '.join(summaries)}"


def visualize_overlap(order):
    # TODO - this function probably doesn't do a very good job estimating the runtime because it doesn't carefully model
    # streams and overlap. For now its mostly useful as a debug visualization.

    total_est_runtime: float = 0.0
    cur_comm_node = None

    def step_log(step, msg):
        overlap_log.debug(f"{step:>6}: {msg}")  # noqa: G004

    for step, snode in enumerate(order):
        if cur_comm_node is None:
            if contains_collective(snode):
                total_est_runtime += estimate_op_runtime(snode)
                cur_comm_node = snode.node
            elif is_wait(snode.node):
                # raise AssertionError(
                #     "Wait is not expected when there is no collective running"
                # )
                pass
            else:  # exposed compute op
                total_est_runtime += estimate_op_runtime(snode)
            step_log(step, f"{node_summary(snode)}")
        else:  # cur_comm_node is not None
            if contains_collective(snode):
                total_est_runtime += estimate_op_runtime(snode)
                cur_comm_node = snode.node
                step_log(step, f"{node_summary(snode)}")  # noqa: G004
            elif is_wait(snode.node):  # end of this comm op
                step_log(step, f"{node_summary(snode)}")
                cur_comm_node = None
            else:  # overlapped compute op
                step_log(step, f"| {node_summary(snode)}")
    overlap_log.debug(
        f"Est. runtime (ms): {total_est_runtime / 1000 / 1000}"  # noqa: G004
    )


def reorder_compute_and_comm_for_overlap(
    snodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    order = snodes
    # pyrefly: ignore [bad-assignment]
    for p in config.reorder_for_compute_comm_overlap_passes:
        if isinstance(p, str) and p in globals():
            p = globals()[p]  # it is a builtin pass
        assert callable(p), (
            f"Invalid reorder_compute_and_comm_for_overlap pass: {p} is not callable"
        )
        order = p(order)  # type: ignore[operator]
    # pyrefly: ignore [bad-return]
    return order


def remove_fsdp2_unsharded_param_graph_input_usage(graph: torch.fx.Graph):
    """
    This FX graph pass replaces uses of FSDP2 unsharded params with their corresponding
    graph intermediates that were fsdp.copy_ into the unsharded params in the original graph.

    NOTE: Can only apply this pass to any of the FSDP2 unsharded params that have this pattern
    (or repetition of): `resize_(full) -> copy_ -> resize_(0)`. Because of this, for partial-graph case
    where `resize_(full) -> copy_` is in one graph and `resize_(0)` is in another graph, we can't
    remove these resize and copy ops and thus we will have worse performance there.

    In other words, "do we try to remove all the resize_(full) -> copy_ -> resize_(0) nodes for this unsharded param"
    is actually a per-unsharded-param decision, since for each unsharded param, we look at its resize sequence pattern
    (in `check_resize_pattern()`) to determine if its set of resize and copy nodes can be removed.
    """
    node_list = list(graph.nodes)

    # Find all graph inputs and their resize counts
    graph_input_to_resized_to_full_node_idxes = defaultdict(list)
    graph_input_to_resized_to_0_node_idxes = defaultdict(list)
    for idx, node in enumerate(node_list):
        if (
            node.op == "call_function"
            and node.target is torch.ops.inductor.resize_storage_bytes_.default
        ):
            assert node.args[0].op == "placeholder", f"""\
Resize can only operate on graph inputs, but got {node} which is resizing non-graph-input {node.args[0]}
"""
            graph_input = node.args[0]
            new_size = node.args[1]
            if new_size > 0:
                graph_input_to_resized_to_full_node_idxes[graph_input].append(idx)
            else:
                graph_input_to_resized_to_0_node_idxes[graph_input].append(idx)

    def check_resize_pattern(graph_input):
        # Check the number of resize-to-full and resize-to-0 nodes are equal,
        # and that for each (resize-to-full, resize-to-0) pair, the resize-to-full node
        # always happens before the resize-to-0 node.
        # This is the precondition for being able to remove all the resize and copy nodes
        # for this specific unsharded param.
        resized_to_full_idxes = graph_input_to_resized_to_full_node_idxes.get(
            graph_input, []
        )
        resized_to_0_idxes = graph_input_to_resized_to_0_node_idxes.get(graph_input, [])

        if len(resized_to_full_idxes) != len(resized_to_0_idxes):
            log.warning(
                f"""
Unequal number of resize-to-full and resize-to-0 nodes for graph input {graph_input}:
{len(resized_to_full_idxes)} vs. {len(resized_to_0_idxes)}.
Skipping `remove_fsdp2_unsharded_param_graph_input_usage` FX graph pass.
"""  # noqa: G004
            )
            return False

        # Check the sequence: (resize_to_full -> resize_to_0)+
        for resize_to_full_idx, resize_to_0_idx in zip(
            resized_to_full_idxes, resized_to_0_idxes
        ):
            if resize_to_full_idx >= resize_to_0_idx:
                log.warning(
                    f"""
For graph input {graph_input}: resize-to-full node {node_list[resize_to_full_idx]} at index {resize_to_full_idx}
happens after resize-to-0 node {node_list[resize_to_0_idx]} at index {resize_to_0_idx}.
Skipping `remove_fsdp2_unsharded_param_graph_input_usage` FX graph pass for that unsharded param.
"""  # noqa: G004
                )
                return False
        return True

    # Find all eligible unsharded params and their corresponding graph intermediates.
    unsharded_param_to_fsdp_copy_node_idxes = defaultdict(list)
    for idx, node in enumerate(node_list):
        if node.op == "call_function" and node.target is torch.ops.fsdp.copy_.default:
            fsdp_copy_node = node
            unsharded_param = node.args[0]
            assert unsharded_param.op == "placeholder", f"""
Assumed all FSDP2 `unsharded_param`s to be graph input, but it's not true!
Offending node: {unsharded_param}. Graph: {graph}
"""
            if check_resize_pattern(unsharded_param):
                unsharded_param_to_fsdp_copy_node_idxes[unsharded_param].append(idx)

    def is_allowed_mutation(node):
        return (
            node.target is torch.ops.fsdp.copy_.default
            or node.target is torch.ops.inductor.resize_storage_bytes_.default
        )

    def is_node_mutating_unsharded_param_or_its_alias(node, unsharded_params):
        # Check whether the node is mutating any of the unsharded params or their aliases.
        mutated_arg_idxes = (
            [
                i
                for i, x in enumerate(node.target._schema.arguments)
                if x.alias_info is not None and x.alias_info.is_write
            ]
            if isinstance(node.target, torch._ops.OpOverload)
            else []
        )
        mutated_node_arg_storages = OrderedSet(
            [
                StorageWeakRef(node.args[i].meta["val"].untyped_storage())
                for i in mutated_arg_idxes
            ]
        )
        storages_of_unsharded_params = OrderedSet(
            [
                StorageWeakRef(unsharded_param.meta["val"].untyped_storage())
                for unsharded_param in unsharded_params
            ]
        )
        return len(mutated_node_arg_storages & storages_of_unsharded_params) > 0

    # Check no user mutation on any unsharded_param
    for node in node_list:
        if (
            node.op == "call_function"
            and isinstance(node.target, torch._ops.OpOverload)
            and node.target._schema.is_mutable
            and not is_allowed_mutation(node)
        ):
            assert not is_node_mutating_unsharded_param_or_its_alias(
                node, unsharded_param_to_fsdp_copy_node_idxes.keys()
            ), f"""\
User mutation on FSDP2 unsharded param is not allowed when Traceable FSDP2 is used. Violating node: {node}
"""

    # For each `fsdp.copy_(unsharded_param, Y)`, replace downstream usage of `unsharded_param` with `Y`.
    #
    # NOTE: Because of "layer reuse" use case, there could be multiple `fsdp.copy_` to the same `unsharded_param` graph input.
    # e.g.
    # ```
    #     fsdp_copy_1 = fsdp.copy_(unsharded_param_1, Y1)
    #     ... (use of unsharded_param_1)                     -> Subgraph 1
    #     fsdp_copy_2 = fsdp.copy_(unsharded_param_1, Y2)
    #     ... (use of unsharded_param_1)                     -> Subgraph 2
    #     fsdp_copy_3 = fsdp.copy_(unsharded_param_1, Y3)
    #     ... (use of unsharded_param_1)                     -> Subgraph 3
    # ```
    # We must do the replacement only within each subgraph.
    for (
        unsharded_param,
        fsdp_copy_node_idxes,
    ) in unsharded_param_to_fsdp_copy_node_idxes.items():
        for i, fsdp_copy_node_idx in enumerate(fsdp_copy_node_idxes):
            fsdp_copy_node = node_list[fsdp_copy_node_idx]
            assert fsdp_copy_node.args[0] is unsharded_param
            _, replacement = fsdp_copy_node.args
            # subgraph_start_idx is exclusive
            subgraph_start_idx = fsdp_copy_node_idx + 1
            # subgraph_end_idx is exclusive (also intentionally don't replace args in return op)
            subgraph_end_idx = (
                fsdp_copy_node_idxes[i + 1]
                if i < len(fsdp_copy_node_idxes) - 1
                else len(node_list) - 1
            )
            subgraph_nodes = node_list[subgraph_start_idx:subgraph_end_idx]
            assert not any(
                is_node_mutating_unsharded_param_or_its_alias(node, [unsharded_param])
                for node in subgraph_nodes
            ), f"""\
Assumed no ops mutating unsharded param {unsharded_param} in subgraph {subgraph_nodes}, but it's not true!
Graph: {graph}
"""
            for node in subgraph_nodes:
                if (
                    node.op == "call_function"
                    and unsharded_param in node.args
                    and node.target != torch.ops.inductor.resize_storage_bytes_.default
                ):  # TODO(yf225): implement replacement in kwargs
                    new_args = tuple(
                        replacement if arg is unsharded_param else arg
                        for arg in node.args
                    )
                    node.args = new_args

    # Delete `fsdp.copy_(unsharded_param, Y)` nodes
    for fsdp_copy_node_idxes in unsharded_param_to_fsdp_copy_node_idxes.values():
        for fsdp_copy_node_idx in fsdp_copy_node_idxes:
            fsdp_copy_node = node_list[fsdp_copy_node_idx]
            graph.erase_node(fsdp_copy_node)

    # Delete `resize_(unsharded_param, ...)` nodes
    for node in node_list:
        if (
            node.op == "call_function"
            and node.target is torch.ops.inductor.resize_storage_bytes_.default
            and node.args[0] in unsharded_param_to_fsdp_copy_node_idxes
        ):
            graph.erase_node(node)


def reinplace_fsdp_all_gather(graph: torch.fx.Graph) -> None:
    try:
        import torch.distributed.fsdp._fully_shard._fsdp_collectives

        assert torch.distributed.is_available()
        # Assert existence of these ops
        assert (
            torch.ops._c10d_functional.all_gather_into_tensor
            and torch.ops._c10d_functional.all_gather_into_tensor_out
        )
    except (ImportError, AttributeError, AssertionError):
        return

    from .pattern_matcher import (
        CallFunction,
        KeywordArg,
        Match,
        PatternMatcherPass,
        register_graph_pattern,
    )

    """
    all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default(...);
    getitem = all_gather_copy_in[0];
    (getitem_1 = all_gather_copy_in[1];)  # optional

    all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem, ...);

    ->

    all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default(...);
    getitem = all_gather_copy_in[0];
    getitem_1 = all_gather_copy_in[1];

    all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor_out.default(getitem, ..., out=getitem_1);
    """

    def remove_unused_getitem(g):
        # Remove `getitem_X = all_gather_copy_in[1]` which is never used.
        node_list = list(g.nodes)
        for n in node_list:
            if (
                n.target is operator.getitem
                and n.args[0].target is torch.ops.fsdp.all_gather_copy_in.default
                and n.args[1] == 1
            ):
                g.erase_node(n)

    graph_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunction(
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            CallFunction(
                operator.getitem,
                CallFunction(
                    torch.ops.fsdp.all_gather_copy_in.default,
                    KeywordArg("all_gather_inputs"),
                    KeywordArg("all_gather_output"),
                    KeywordArg("inp_split_sizes"),
                    KeywordArg("all_gather_input_numel"),
                    KeywordArg("rank"),
                ),
                KeywordArg("item_idx"),
            ),
            KeywordArg("group_size"),
            KeywordArg("group_name"),
        ),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=graph_pass,
        extra_check=lambda match: match.kwargs["item_idx"] == 0,
    )
    def reinplace_all_gather(match: Match, *args, **kwargs):
        def repl(
            *args,
        ):
            copy_in_args = args[:-2]
            group_size = args[-2]
            group_name = args[-1]
            all_gather_copy_in = torch.ops.fsdp.all_gather_copy_in.default(
                *copy_in_args
            )
            getitem = all_gather_copy_in[0]
            getitem_1 = all_gather_copy_in[1]
            all_gather_into_tensor = (
                torch.ops._c10d_functional.all_gather_into_tensor_out.default(
                    getitem, group_size, group_name, out=getitem_1
                )
            )
            return all_gather_into_tensor

        match.replace_by_example(
            # pyrefly: ignore [bad-argument-type]
            repl,
            [
                kwargs["all_gather_inputs"],
                kwargs["all_gather_output"],
                kwargs["inp_split_sizes"],
                kwargs["all_gather_input_numel"],
                kwargs["rank"],
                kwargs["group_size"],
                kwargs["group_name"],
            ],
        )

    remove_unused_getitem(graph)
    graph_pass.apply(graph)  # type: ignore[arg-type]


def get_op_idx(snode):
    assert not isinstance(
        snode,
        (
            torch._inductor.scheduler.FusedSchedulerNode,
            torch._inductor.scheduler.GroupedSchedulerNode,
        ),
    )
    return int(snode.get_name()[2:])


def enforce_comm_ordering_for_fsdp(
    snodes: list[torch._inductor.scheduler.BaseSchedulerNode],
    name_to_buf: dict[str, torch._inductor.scheduler.SchedulerBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
) -> list[torch._inductor.scheduler.BaseSchedulerNode]:
    from . import scheduler

    new_order: list[BaseSchedulerNode] = []
    scheduled = OrderedSet[Any]()
    ag_exists = False
    rs_exists = False
    ag_grouped_node_to_wait_grouped_node = {}
    rs_grouped_node_to_wait_grouped_node = {}
    snode_name_to_final_snode = {}

    def _create_group_node(snodes_to_group):
        group_node = scheduler.GroupedSchedulerNode.create(snodes_to_group)
        for snode in snodes_to_group:
            snode_name_to_final_snode[snode.get_name()] = group_node
        snode_name_to_final_snode[group_node.get_name()] = group_node
        return group_node

    # Create grouped nodes for specific sets of ops
    for snode in snodes:
        # Case 1: Handle AllGather
        if is_collective(
            snode.node, op=torch.ops._c10d_functional.all_gather_into_tensor_out.default
        ) and any(
            is_fallback_op(
                name_to_fused_node[x].node, torch.ops.fsdp.all_gather_copy_in.default
            )
            for x in snode.ancestors
        ):
            ag_exists = True
            ag_snode = snode
            ag_related_snode_set: OrderedSet[scheduler.BaseSchedulerNode] = OrderedSet()

            # Find the "cast + copy_in + getitem + all_gather" code block
            find_recursive_deps_of_node(
                ag_snode,
                ag_related_snode_set,
                name_to_buf,
                name_to_fused_node,
            )

            # Find the "all_gather + all_gather_wait_tensor + copy_out" code block
            allowed_ops = OrderedSet(
                [
                    torch.ops._c10d_functional.all_gather_into_tensor_out.default,
                    torch.ops._c10d_functional.wait_tensor.default,
                    torch.ops.fsdp.split_with_sizes_copy.default,
                ]
            )
            find_recursive_users_of_node(
                ag_snode,
                ag_related_snode_set,
                name_to_buf,
                name_to_fused_node,
                criteria_cb=lambda x: not (
                    isinstance(x, scheduler.NopKernelSchedulerNode)
                    or (
                        isinstance(x, scheduler.ExternKernelSchedulerNode)
                        and x.node.op_overload in allowed_ops  # type: ignore[union-attr]
                    )
                ),
            )

            # sort nodes by original operation order
            ag_related_snodes = sorted(
                ag_related_snode_set, key=lambda x: get_op_idx(x)
            )

            # In the "reuse layer" case, some ops in the 2nd all-gather code block could also
            # depend on ops in the 1st all-gather code block, and we don't want to group them together.
            end_idx_of_current_ag_block = len(ag_related_snodes)
            copy_out_count = 0
            for i in range(len(ag_related_snodes)):
                cur_snode = ag_related_snodes[i]
                if is_fallback_op(
                    cur_snode.node, torch.ops.fsdp.split_with_sizes_copy.default
                ):
                    copy_out_count += 1
                if copy_out_count > 1:
                    end_idx_of_current_ag_block = i
                    break

            ag_related_snodes = ag_related_snodes[:end_idx_of_current_ag_block]

            # Group "cast + copy_in + getitem + all_gather" into one GroupedSchedulerNode
            wait_node_idx = None
            for i in range(len(ag_related_snodes) - 1):
                if isinstance(ag_related_snodes[i + 1].node, ir._WaitKernel):
                    wait_node_idx = i + 1
                    break
            assert wait_node_idx is not None
            ag_group_node = _create_group_node(ag_related_snodes[:wait_node_idx])

            # Group "all_gather_wait_tensor + copy_out" into one GroupedSchedulerNode
            ag_wait_group_node = _create_group_node(ag_related_snodes[wait_node_idx:])

            ag_grouped_node_to_wait_grouped_node[ag_group_node] = ag_wait_group_node

        # Case 2: Handle ReduceScatter
        elif is_fallback_op(snode.node, torch.ops.fsdp.chunk_cat.default):
            rs_exists = True
            rs_snode = snode

            # Find the "reduce_scatter copy-in + reduce_scatter comm + reduce_scatter wait" code block
            rs_related_snode_set: OrderedSet[scheduler.BaseSchedulerNode] = OrderedSet()
            find_recursive_users_of_node(
                rs_snode,
                rs_related_snode_set,
                name_to_buf,
                name_to_fused_node,
            )

            # sort nodes by original operation order
            rs_related_snodes = sorted(
                rs_related_snode_set, key=lambda x: get_op_idx(x)
            )

            # Group "reduce_scatter copy-in + reduce_scatter comm" into one GroupedSchedulerNode
            wait_node_idx = None
            for i in range(len(rs_related_snodes) - 1):
                if isinstance(rs_related_snodes[i + 1].node, ir._WaitKernel):
                    wait_node_idx = i + 1
                    break
            assert wait_node_idx is not None
            rs_group_node = _create_group_node(rs_related_snodes[:wait_node_idx])

            # Group "reduce_scatter wait + related output nodes" into one GroupedSchedulerNode
            rs_wait_group_node = _create_group_node(rs_related_snodes[wait_node_idx:])

            rs_grouped_node_to_wait_grouped_node[rs_group_node] = rs_wait_group_node

    assert len(snode_name_to_final_snode) > 0
    if ag_exists:
        assert len(ag_grouped_node_to_wait_grouped_node) > 0
    if rs_exists:
        assert len(rs_grouped_node_to_wait_grouped_node) > 0

    # Build the new node schedule, taking GroupedSchedulerNode into account
    for snode in snodes:
        if snode.get_name() in snode_name_to_final_snode:
            snode = snode_name_to_final_snode[snode.get_name()]
        if snode in scheduled:
            continue
        new_order.append(snode)
        scheduled.add(snode)

    # Enforce AllGather ordering: previous AllGather's "wait then copy_out" group node must run
    # before next AllGather's "copy_in then AG" group node
    prev_ag_wait = None
    for ag_group_node, wait_group_node in ag_grouped_node_to_wait_grouped_node.items():
        if prev_ag_wait is not None:
            mutating_buf = next(iter(ag_group_node.get_buffer_names()))
            for o in prev_ag_wait.get_outputs():
                ag_group_node.add_fake_dep(
                    WeakDep(o.get_name(), mutating_buf=mutating_buf, is_fake=True)
                )
        prev_ag_wait = wait_group_node

    # Enforce ReduceScatter ordering: previous ReduceScatter's "wait" group node must run
    # before next ReduceScatter's "copy_in then RS" group node
    prev_rs_wait = None
    for rs_group_node, wait_group_node in rs_grouped_node_to_wait_grouped_node.items():
        if prev_rs_wait is not None:
            mutating_buf = next(iter(rs_group_node.get_buffer_names()))
            for o in prev_rs_wait.get_outputs():
                rs_group_node.add_fake_dep(
                    WeakDep(o.get_name(), mutating_buf=mutating_buf, is_fake=True)
                )
        prev_rs_wait = wait_group_node

    return new_order  # type: ignore[return-value]
