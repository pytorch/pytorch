# mypy: allow-untyped-defs
# pyre-strict
from __future__ import annotations

import heapq
import importlib
import logging
import operator
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
from torch._logging import trace_structured
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._ordered_set import OrderedSet

from . import config, ir
from .dependencies import WeakDep


if TYPE_CHECKING:
    from .ir import IRNode, Operation

from .memory import estimate_peak_memory, FreeableInputBuffer, get_freeable_input_buf
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

    initial_exposed: float = -1
    final_exposed: float = -1
    limiting_factor: str = "None"
    moves: int = 0
    grouped: int = 0
    grouped_info: str = ""

    @property
    def improvement(self):
        return self.initial_exposed - self.final_exposed


def is_gemm_like(node: Optional[Union[IRNode, Operation]]) -> bool:
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


def _temp_group_visit_leaves(snode, fn):
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


def _reorder_communication_preserving_peak_memory_internal(
    snodes: list[BaseSchedulerNode],
) -> tuple[list[BaseSchedulerNode], dict[BaseSchedulerNode, ReorderInfo]]:
    from torch._inductor.scheduler import GroupedSchedulerNode

    original_snodes_num = len(snodes)
    """
    Internal testing helper that also returns debug info.
    Returns:
        - reordered snodes list
        - dict {snode: ReorderInfo}
    """
    # heuristic to avoid degenerating to quadratic time
    graph_inputs: OrderedSet[str] = OrderedSet(V.graph.graph_inputs.keys())
    graph_outputs: OrderedSet[str] = OrderedSet(V.graph.get_output_names())
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer] = get_freeable_input_buf(
        snodes, graph_inputs
    )
    peak_memory, curr_memory = estimate_peak_memory(
        snodes, name_to_freeable_input_buf, graph_outputs
    )
    runtimes = {snode: estimate_op_runtime(snode) for snode in snodes}
    _curr_memory = dict(zip(snodes, curr_memory))
    _curr_memory[None] = 0  # type: ignore[index]

    # debug stats
    stats: dict[BaseSchedulerNode, ReorderInfo] = {}

    def exposed_communication_time(collective_snode, remaining_snodes):
        # assumes a linear schedule and computes the overlap of the collective with the remaining nodes
        comm_time = estimate_op_runtime(collective_snode)
        compute_time = 0.0
        for snode in remaining_snodes:
            if contains_collective(snode):
                continue
            if contains_wait(snode):
                # TODO - if the wait is for a collective that started before this collective or on another stream,
                # we can ignore it. Otherwise, it's the end of the road for overlap opportunities
                break

            def accumulate_time(_snode):
                nonlocal compute_time
                compute_time += runtimes[_snode]

            _temp_group_visit_leaves(snode, accumulate_time)
        return max(0, comm_time - compute_time)

    total_moves = 0

    # Dicts to keep track of "next" and "previous" as double-linked structure during grouping
    _prev: dict[Optional[BaseSchedulerNode], Optional[BaseSchedulerNode]] = {}
    _next: dict[Optional[BaseSchedulerNode], Optional[BaseSchedulerNode]] = {}
    for i, snode in enumerate(snodes):
        _prev[snode] = snodes[i - 1] if i > 0 else None
        _next[snode] = snodes[i + 1] if i < len(snodes) - 1 else None
    _curr_memory = dict(zip(snodes, curr_memory))
    _curr_memory[None] = 0  # type: ignore[index]

    _head = snodes[0]

    def _group_nodes(head, tail):
        ret = []
        n = head
        while True:
            if n is not None:
                ret.append(n)
            if n == tail:
                break
            n = _next[n]
        return ret

    def _group_names(head, tail):
        ret = ""
        for n in _group_nodes(head, tail):
            if ret:
                ret += "~"
            ret += n.get_name()
        return ret

    curr = _head
    while _next[curr] is not None:
        if contains_collective(curr):
            reorder_info = stats[curr] = ReorderInfo()
            reorder_info.initial_exposed = reorder_info.final_exposed = (
                exposed_communication_time(curr, _group_nodes(_next[curr], None))
            )

            candidate = _prev[curr]
            group_head = curr
            group_tail = curr
            group_peak_memory = _curr_memory[curr]
            while candidate is not None:
                if contains_collective(candidate):
                    reorder_info.limiting_factor = "collective ordering"
                    break

                group = GroupedSchedulerNode(
                    curr.scheduler,
                    _group_nodes(group_head, group_tail),
                    temp_grouping=True,
                )

                data_deps = {s.name: s for s in group.unmet_dependencies}
                candidate_outs = candidate.get_outputs()
                data_dep = None
                for o in candidate_outs:
                    if d := data_deps.get(o.get_name(), None):
                        if isinstance(d, WeakDep) and d.is_fake:
                            continue
                        data_dep = d
                        break

                if data_dep is not None:

                    def is_groupable(candidate):
                        # preserve ordering
                        if contains_collective(candidate):
                            return False, "contains_collective"

                        if contains_gemm_like(candidate):
                            return False, "contains_gemm_like"
                        return True, None

                    is_grp, grp_reason = is_groupable(candidate)
                    if is_grp:
                        group_head = candidate
                        group_peak_memory = max(
                            group_peak_memory, _curr_memory[candidate]
                        )
                        reorder_info.grouped += 1
                        reorder_info.grouped_info = _group_names(group_head, group_tail)
                        candidate = _prev[candidate]
                        continue
                    else:
                        msg = (
                            f"data dependency {data_dep}(dep_names:{list(data_deps.keys())})"
                            f"\n candidate:{candidate.get_name()}(os:{[candidate.get_buffer_names()]})"
                            f"dep on {_group_names(group_head, group_tail)}"
                            f"\n non_group_reason:{grp_reason}"
                        )
                        reorder_info.limiting_factor = msg
                        break

                delta_memory_candidate = (
                    _curr_memory[candidate] - _curr_memory[_prev[candidate]]  # type: ignore[index]
                )

                if group_peak_memory - delta_memory_candidate > peak_memory:
                    reorder_info.limiting_factor = "peak memory"
                    break

                reorder_info.moves += 1
                total_moves += 1

                mem_deltas = {}
                for n in [candidate, *_group_nodes(group_head, group_tail)]:
                    mem_deltas[n] = _curr_memory[n] - _curr_memory[_prev[n]]  # type: ignore[index]
                # swap (candidate, group_head...group_tail)
                # Before:
                # candidate_prev -0-> candidate -1-> group_head...group_tail -2-> group_tail_next
                # After:
                # candidate_prev -0-> group_head...group_tail -1-> candidate -2-> group_tail_next
                # 0
                candidate_prev = _prev[candidate]
                if candidate_prev:
                    _next[candidate_prev] = group_head
                _prev[group_head] = candidate_prev

                # 2
                group_tail_next = _next[group_tail]
                if group_tail_next:
                    _prev[group_tail_next] = candidate
                _next[candidate] = group_tail_next

                # 1
                _prev[candidate] = group_tail
                _next[group_tail] = candidate

                if _head == candidate:
                    _head = group_head

                reorder_info.final_exposed = exposed_communication_time(
                    curr, _group_nodes(_next[curr], None)
                )
                # Recompute curr_memory
                _prev_curr_memory = _curr_memory[_prev[group_head]]  # type: ignore[index]
                for n in _group_nodes(group_head, candidate):
                    _curr_memory[n] = _prev_curr_memory = (
                        _prev_curr_memory + mem_deltas[n]
                    )
                candidate = _prev[group_head]
        curr = _next[curr]  # type: ignore[assignment]

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
        "initial exposed",
        "final exposed",
        "improvement",
        "limiting factor",
        "moves",
        "grouped",
        "grouped_info",
    ]
    rows = [
        [
            node_summary(snode),
            node_reorder_info.initial_exposed,
            node_reorder_info.final_exposed,
            node_reorder_info.improvement,
            node_reorder_info.limiting_factor,
            node_reorder_info.moves,
            node_reorder_info.grouped,
            node_reorder_info.grouped_info,
        ]
        for snode, node_reorder_info in node_stats.items()
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

    new_snodes = _group_nodes(_head, None)
    assert len(new_snodes) == original_snodes_num
    new_peak_memory, curr_memory = estimate_peak_memory(
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

    while len(ready):
        snode = heapq.heappop(ready).snode
        if reorder_for_overlap and contains_collective(snode):
            schedule_collective_for_overlap(snode)
        else:
            schedule(snode)

    for snode, deps in unmet_deps.items():
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


def _sink_waits_iterative_internal(
    snodes: list[BaseSchedulerNode],
) -> tuple[list[BaseSchedulerNode], dict[BaseSchedulerNode, SinkWaitInfo]]:
    from torch._inductor.scheduler import GroupedSchedulerNode

    original_snodes_num = len(snodes)
    if original_snodes_num == 0:
        return snodes, {}
    graph_inputs: OrderedSet[str] = OrderedSet(V.graph.graph_inputs.keys())
    graph_outputs: OrderedSet[str] = OrderedSet(V.graph.get_output_names())
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer] = get_freeable_input_buf(
        snodes, graph_inputs
    )
    peak_memory, curr_memory = estimate_peak_memory(
        snodes, name_to_freeable_input_buf, graph_outputs
    )

    stats: dict[BaseSchedulerNode, SinkWaitInfo] = {}
    _prev: dict[Optional[BaseSchedulerNode], Optional[BaseSchedulerNode]] = {}
    _next: dict[Optional[BaseSchedulerNode], Optional[BaseSchedulerNode]] = {}
    _head = snodes[0]
    for i, snode in enumerate(snodes):
        _prev[snode] = snodes[i - 1] if i > 0 else None
        _next[snode] = snodes[i + 1] if i < len(snodes) - 1 else None
    _curr_memory = dict(zip(snodes, curr_memory))
    _curr_memory[None] = 0  # type: ignore[index]

    def _group_nodes(head, tail):
        ret = []
        n = head
        while True:
            if n is not None:
                ret.append(n)
            if n == tail:
                break
            n = _next[n]
        return ret

    def _group_names(head, tail):
        ret = ""
        for n in _group_nodes(head, tail):
            if ret:
                ret += "~"
            ret += n.get_name()
        return ret

    curr = snodes[-1]

    processed_waits = OrderedSet()  # type: ignore[var-annotated]
    swap_i = 0
    while _prev[curr] is not None:
        if contains_wait(curr) and curr not in processed_waits:
            processed_waits.add(curr)
            info = stats[curr] = SinkWaitInfo()
            candidate = _next[curr]
            wait_snode = curr
            group_head = curr
            group_tail = curr
            group_peak_memory = _curr_memory[curr]
            while candidate is not None:
                group = GroupedSchedulerNode(
                    wait_snode.scheduler,
                    _group_nodes(group_head, group_tail),
                    temp_grouping=True,
                )
                group_outs = group.get_outputs()

                data_deps = {s.name: s for s in candidate.unmet_dependencies}
                data_dep = None
                for o in group_outs:
                    if d := data_deps.get(o.get_name(), None):
                        if isinstance(d, WeakDep) and d.is_fake:
                            continue
                        data_dep = d
                        break
                # 1. If we have data_dep - we can not swap => trying to group
                # 2. If swap candidate and current node both contain collectives => trying to group
                if data_dep is not None or (
                    both_contain_comms := (
                        contains_collective(group) and contains_collective(candidate)
                    )
                ):

                    def is_groupable(snode):
                        # We do not want to group with collectives to not reorder them forward.
                        if contains_collective(snode):
                            return (
                                False,
                                f"candidate contains collective {snode.get_name()}",
                            )
                        if contains_gemm_like(snode):
                            return (
                                False,
                                f"candidate contains gemm_like {snode.get_name()}",
                            )
                        return True, None

                    is_grp, grp_reason = is_groupable(candidate)
                    if is_grp:
                        group_tail = candidate
                        group_peak_memory = max(
                            group_peak_memory, _curr_memory[candidate]
                        )
                        info.grouped += 1
                        info.grouped_info = _group_names(group_head, group_tail)
                        candidate = _next[candidate]
                        continue
                    elif (data_dep is None) and both_contain_comms:
                        info.limiting_factor = (
                            f"collective ordering {_group_names(group_head, group_tail)}"
                            f" with candidate:{candidate.get_name()}"
                        )
                        break
                    else:
                        info.limiting_factor = (
                            f"data dependency {data_dep}(dep_names:{list(data_deps.keys())})"
                            f"\n candidate:{candidate.get_name()}(os:{[candidate.get_buffer_names()]})"
                            f"dep on {_group_names(group_head, group_tail)}"
                            f"\n outs:{[o.get_name() for o in group_outs]}"
                            f"\n non_group_reason:{grp_reason}"
                        )
                        break
                candidate_delta_memory = (
                    _curr_memory[candidate] - _curr_memory[_prev[candidate]]  # type: ignore[index]
                )
                if group_peak_memory + candidate_delta_memory > peak_memory:
                    info.limiting_factor = "peak_memory"
                    break

                info.moves += 1
                info.moves_info += f"+{candidate.get_name()}"
                print(f"XXX SWAP {candidate.get_name()} vs {_group_names(group_head, group_tail)}")

                # group_head_prev -0-> candidate -1-> group_head...group_tail -2-> candidate_next
                mem_deltas = {}
                for n in [candidate, *_group_nodes(group_head, group_tail)]:
                    mem_deltas[n] = _curr_memory[n] - _curr_memory[_prev[n]]  # type: ignore[index]
                # 0:
                group_head_prev = _prev[group_head]
                if group_head_prev:
                    _next[group_head_prev] = candidate
                _prev[candidate] = group_head_prev

                # 2:
                candidate_next = _next[candidate]
                if candidate_next:
                    _prev[candidate_next] = group_tail
                _next[group_tail] = candidate_next

                # 1:
                _prev[group_head] = candidate
                _next[candidate] = group_head
                if group_head == _head:
                    _head = candidate

                # Recompute curr_memory
                _prev_curr_memory = _curr_memory[_prev[candidate]]  # type: ignore[index]
                for n in _group_nodes(candidate, group_tail):
                    _curr_memory[n] = _prev_curr_memory = (
                        _prev_curr_memory + mem_deltas[n]
                    )

                ### DEBUG_PEAK
                new_snodes = _group_nodes(_head, None)
                new_peak_memory, new_curr_memory = estimate_peak_memory(
                    new_snodes, name_to_freeable_input_buf, graph_outputs
                )
                print(f"XXX SWAP {swap_i}")
                print(f"XXX NEW_PEAK:{new_peak_memory} old_peak:{peak_memory}")
                for i, node in enumerate(new_snodes):
                    print(f"XXX {node.get_name()} CURR:{new_curr_memory[i]} ITER_CURR:{_curr_memory[node]}")
                swap_i += 1
                if new_peak_memory > peak_memory:
                    assert False
                ### DEBUG_PEAK
                candidate = _next[group_tail]
        curr = _prev[curr]  # type: ignore[assignment]

    headers = [
        "Wait node",
        "grouped",
        "grouped_info",
        "moves",
        "moves_info",
        "limiting factor",
    ]
    rows = [
        [
            node_summary(snode),
            info.grouped,
            info.grouped_info,
            info.moves,
            info.moves_info,
            info.limiting_factor,
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
    new_snodes = _group_nodes(_head, None)
    assert len(new_snodes) == original_snodes_num
    new_peak_memory, curr_memory = estimate_peak_memory(
        new_snodes, name_to_freeable_input_buf, graph_outputs
    )
    log_str += f"\n peak_memory_before:{peak_memory}"
    log_str += f"\n peak_memory_after:{new_peak_memory}"
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "sink_waits_iterative_info",
            "encoding": "string",
        },
        payload_fn=lambda: log_str,
    )
    return new_snodes, stats


def sink_waits_iterative(
    snodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    return _sink_waits_iterative_internal(snodes)[0]


def estimate_op_runtime(snode: BaseSchedulerNode) -> float:
    """
    Returns estimated op runtime in nanoseconds (ns)
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
            detail = f" {snode.get_name()} ({snode.node.python_kernel_name})\n {outs_str}\n ({ins_str})"
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
    graph_inputs: OrderedSet[str] = OrderedSet(V.graph.graph_inputs.keys())
    graph_outputs: OrderedSet[str] = OrderedSet(V.graph.get_output_names())
    for p in config.reorder_for_compute_comm_overlap_passes:
        if isinstance(p, str) and p in globals():
            p = globals()[p]  # it is a builtin pass
        assert callable(p), (
            f"Invalid reorder_compute_and_comm_for_overlap pass: {p} is not callable"
        )
        peak_memory, _ = estimate_peak_memory(
            snodes, get_freeable_input_buf(snodes, graph_inputs), graph_outputs
        )
        if torch.distributed.get_rank() == 0:
            overlap_log.debug(
                f"==== Visualize overlap before reordering pass {p}, {peak_memory=} ===="  # noqa: G004
            )
            try:
                visualize_overlap(order)
            except Exception as e:
                overlap_log.debug("", exc_info=e)
        t0 = time.time()
        order = p(order)  # type: ignore[operator]
        t = time.time() - t0
        if torch.distributed.get_rank() == 0:
            overlap_log.debug(
                f"==== Visualize overlap after reordering pass {p} (ran in {t} sec)===="  # noqa: G004
            )
            try:
                visualize_overlap(order)
            except Exception as e:
                overlap_log.debug("", exc_info=e)
        peak_memory, _ = estimate_peak_memory(
            snodes, get_freeable_input_buf(snodes, graph_inputs), graph_outputs
        )
        print(f"final {peak_memory=}")
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
            and node.target == torch.ops.inductor.resize_storage_bytes_.default
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

        if not len(resized_to_full_idxes) == len(resized_to_0_idxes):
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
        if node.op == "call_function" and node.target == torch.ops.fsdp.copy_.default:
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
            node.target == torch.ops.fsdp.copy_.default
            or node.target == torch.ops.inductor.resize_storage_bytes_.default
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
    for (
        unsharded_param,
        fsdp_copy_node_idxes,
    ) in unsharded_param_to_fsdp_copy_node_idxes.items():
        for i, fsdp_copy_node_idx in enumerate(fsdp_copy_node_idxes):
            fsdp_copy_node = node_list[fsdp_copy_node_idx]
            graph.erase_node(fsdp_copy_node)

    # Delete `resize_(unsharded_param, ...)` nodes
    for node in node_list:
        if (
            node.op == "call_function"
            and node.target == torch.ops.inductor.resize_storage_bytes_.default
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
                n.target == operator.getitem
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
