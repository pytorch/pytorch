"""
TODO(yf225):
1. add unit tests for this file (e.g. simple graphs where we know what the optimal ordering looks like as ground truth)
"""

# pyre-strict

import os
from typing import List

import torch.fx as fx

from . import ir, scheduler
from .analysis import get_runtime_of_snode
from .utils import printd
from .dependencies import WeakDep


# Used to ensure that iterating over a set is deterministic
def tuple_sorted(x):
    if len(x) == 0:
        return []

    def sort_func(elem):
        if isinstance(elem, str):
            return elem
        else:
            # We expect `elem` to be `scheduler.BaseSchedulerNode` type here,
            # but we are not able to do isinstance assert because of circular dependency
            return elem.get_name()

    return sorted(x, key=sort_func)


def sink_waits(result: List["scheduler.BaseSchedulerNode"]) -> List["scheduler.BaseSchedulerNode"]:
    """
    Greedily moves waits as late as possible (i.e. until we reach a use). Optimal in terms of
    communication overlap.
    """
    new_result = []
    cur_waits = set()
    for snode in result:
        if isinstance(snode.node, ir.Wait):
            cur_waits.add(snode)
        else:
            for wait in tuple_sorted(cur_waits):
                if snode in wait.node_users:
                    new_result.append(wait)
                    cur_waits.remove(wait)
            new_result.append(snode)
    for snode in tuple_sorted(cur_waits):
        new_result.append(snode)
    return new_result


def raise_comms(result: List["scheduler.BaseSchedulerNode"]) -> List["scheduler.BaseSchedulerNode"]:
    """
    Greedily moves comms as early as possible (i.e. until we reach an input).
    Optimal in terms of communication overlap.
    """
    new_result: List["scheduler.BaseSchedulerNode"] = []
    cur_comms: List["scheduler.BaseSchedulerNode"] = []
    for snode in reversed(result):
        if isinstance(snode.node, ir.CollectiveKernel):
            cur_comms.append(snode)
        else:
            for comm in cur_comms:
                assert len(comm.args) > 0
            while len(cur_comms) > 0 and any([snode in comm.args for comm in cur_comms]):
                comm = cur_comms.pop(0)
                new_result.append(comm)
            new_result.append(snode)
    assert len(cur_comms) <= 1
    for snode in tuple_sorted(cur_comms):
        new_result.append(snode)
    result = new_result[::-1]
    return result


def get_ancestors(node):
    ancestors = set()
    cur_nodes = [node]
    while len(cur_nodes) > 0:
        new_nodes = []
        for node in cur_nodes:
            for inp in node.args:
                if inp not in ancestors:
                    ancestors.add(inp)
                    new_nodes.append(inp)
        cur_nodes = new_nodes
    return ancestors


def get_descendants(node):
    descendants = set()
    cur_nodes = [node]
    while len(cur_nodes) > 0:
        new_nodes = []
        for node in cur_nodes:
            for inp in node.node_users:
                if inp not in descendants:
                    descendants.add(inp)
                    new_nodes.append(inp)
        cur_nodes = new_nodes
    return descendants


def decide_global_ordering_of_comms(nodes: List["scheduler.BaseSchedulerNode"]):
    """
    Decide global ordering of comms, by just enforcing the ordering that's in the input graph.
    TODO: Come up with a better approach
    """
    comm_nodes = [n for n in nodes if isinstance(n.node, ir.CollectiveKernel)]
    for i in range(1, len(comm_nodes)):
        # Enforce ordering by making previous comm a `WeakDep` dependency of the next comm
        comm_nodes[i].add_mutation_dep(WeakDep(comm_nodes[i - 1].get_name()))


def assert_no_comm_nodes(snodes: List["scheduler.BaseSchedulerNode"]) -> None:
    assert not any(isinstance(snode.node, ir.CollectiveKernel) for snode in snodes)


def reorder_compute_and_comm_for_overlap(snodes: List["scheduler.BaseSchedulerNode"]) -> List["scheduler.BaseSchedulerNode"]:
    """
    Decides a global ordering of all compute and communication nodes. Assumes that we already have a global ordering of communication nodes.
    Overall procedure is:
        Priority 1. Given that we've currently scheduled comm N, we now schedule all compute nodes that are required for comm N + 1 but do not depend on comm N, to run at the same time with comm N.
        Priority 2. Now, if all those compute nodes are sufficient to overlap comm N, we're done. Otherwise, we now need to look elsewhere to find compute that overlaps with comm N. We prioritize compute nodes that are needed sooner.
        Priority 3. Now, we schedule the compute nodes dependent on comm N and required for comm N + 1.
        Repeat this procedure for subsequent comm nodes.
    """

    comm_nodes = []
    for snode in snodes:
        if isinstance(snode.node, ir.CollectiveKernel):
            comm_nodes.append(snode)

    comm_ancestors = {node: get_ancestors(node) for node in comm_nodes}
    comm_descendants = {node: get_descendants(node) for node in comm_nodes}

    indeg = {k: 0 for k in snodes}
    for snode in snodes:
        for user in snode.node_users:
            if user in indeg:
                indeg[user] += 1
    ready_to_schedule_nodes = set([node for node in snodes if indeg[node] == 0])

    unscheduled_nodes = set()
    unscheduled_nodes = set([snode for snode in snodes])

    result: List["scheduler.BaseSchedulerNode"] = []

    def schedule_node(snode):
        """
        Schedule a single node.
        """
        assert snode in unscheduled_nodes
        assert snode in ready_to_schedule_nodes
        ready_to_schedule_nodes.remove(snode)
        unscheduled_nodes.remove(snode)
        result.append(snode)
        for user in tuple_sorted(snode.node_users):
            if user in indeg:
                indeg[user] -= 1
                if indeg[user] == 0:
                    ready_to_schedule_nodes.add(user)

    def schedule_nodes(snodes):
        """
        Schedules all nodes in `snodes` in an arbitrary topologically valid order.
        """
        all_nodes = set(snodes)
        assert all([node in unscheduled_nodes for node in all_nodes])
        while len(all_nodes) > 0:
            # NOTE: since model graph is always a DAG and does not have circular dependency inside,
            # there should be at least one node that is a "free node" (i.e. indeg == 0),
            # hence infinite loop is not possible. But we check here just to be safe.
            progress = False
            for node in tuple_sorted(all_nodes):
                if node in ready_to_schedule_nodes:
                    schedule_node(node)
                    all_nodes.remove(node)
                    progress = True
            if not progress:
                raise Exception("Unable to find a free node (indeg == 0). This is an impossible state to reach. Please report a bug to PyTorch.")

    # First, schedule all compute nodes that are required by first comm node,
    # as well as the first comm node itself.
    result = schedule_nodes(
        result,
        list(comm_ancestors[comm_nodes[0]]) + [comm_nodes[0]],
    )

    rolled_over_compute_cost = 0
    for idx in range(1, len(comm_ancestors)):
        # Priority 1: Given that we've currently scheduled comm `idx-1`, we now schedule
        # all compute nodes that are required for comm `idx` but do not depend on comm `idx-1`,
        # to run at the same time with comm `idx-1`.
        needed_by_comm_N_and_ready_compute_nodes = unscheduled_nodes & (
            comm_ancestors[comm_nodes[idx]] - comm_descendants[comm_nodes[idx - 1]]
        )
        assert_no_comm_nodes(needed_by_comm_N_and_ready_compute_nodes)

        total_compute_runtime_cost = rolled_over_compute_cost + sum(
            [get_runtime_of_snode(node) for node in needed_by_comm_N_and_ready_compute_nodes]
        )
        prev_comm_runtime_cost = get_runtime_of_snode(comm_nodes[idx - 1])
        schedule_nodes(tuple_sorted(needed_by_comm_N_and_ready_compute_nodes))

        # Priority 2: Now, if all those compute nodes are sufficient to overlap comm `idx-1`, we're done.
        # Otherwise, we now need to look elsewhere to find compute that overlaps with comm `idx`.
        # We prioritize compute nodes that are needed sooner.
        priority1_runtime_cost = total_compute_runtime_cost
        if priority1_runtime_cost >= prev_comm_runtime_cost:
            pass
        else:
            # Find all ready to schedule compute nodes that do not depend on comm `idx-1`.
            ready_to_schedule_compute_nodes = tuple_sorted(
                ready_to_schedule_nodes - comm_descendants[comm_nodes[idx - 1]]
            )
            assert_no_comm_nodes(ready_to_schedule_compute_nodes)

            def earliest_comm_descendant(node):
                for idx in range(len(comm_nodes)):
                    if node in comm_ancestors[comm_nodes[idx]]:
                        return idx
                return len(comm_nodes)

            # Prioritize compute nodes that are needed sooner.
            ready_to_schedule_compute_nodes = sorted(
                ready_to_schedule_compute_nodes, key=earliest_comm_descendant
            )

            for snode in ready_to_schedule_compute_nodes:
                if total_compute_runtime_cost >= prev_comm_runtime_cost:
                    # If accumulated compute runtime cost is greater than comm `idx-1` runtime cost,
                    # it means we have maximized overlap for comm `idx-1`, and hence we stop looking
                    # for more compute to schedule.
                    break
                compute_runtime_cost = get_runtime_of_snode(snode)
                # If we're not able to leverage more than half of this
                # node's compute to overlap, we skip it.
                # TODO: Smarter heuristics here
                if (prev_comm_runtime_cost - total_compute_runtime_cost) <= compute_runtime_cost / 2:
                    continue
                schedule_node(snode)
                total_compute_runtime_cost += compute_runtime_cost
        rollable_compute_cost = total_compute_runtime_cost - priority1_runtime_cost

        # Priority 3. Now, we schedule the compute nodes dependent on comm N and required for comm N + 1.
        is_comm_blocking = (
            len(comm_descendants[comm_nodes[idx - 1]] & comm_ancestors[comm_nodes[idx]])
            > 0
        )
        # The idea here is that if there are no compute nodes from Priority 3, we
        # can roll over the compute nodes in Priority 2 to overlap with the next comm, since
        # they're not required to finish before the next comm starts.
        # TODO: We can extend our ability to roll over compute if we leverage low
        # priority streams here, since that would alleviate us from the requirement
        # to finish Priority 2 compute before the next comm starts.
        if is_comm_blocking:
            rolled_over_compute_cost = 0
        else:
            rolled_over_compute_cost = rollable_compute_cost

        priority3 = unscheduled_nodes & comm_ancestors[comm_nodes[idx]]
        schedule_nodes(list(priority3) + [comm_nodes[idx]])

        debug_print()

    schedule_nodes(unscheduled_nodes)

    result = sink_waits(result)
    result = raise_comms(result)
    return result
