"""
TODO(yf225):
1. add unit tests for this file (e.g. simple graphs where we know what the optimal ordering looks like as ground truth)
"""

# pyre-strict

import os
from typing import List

import torch.fx as fx

from . import ir, scheduler
from .analysis import get_runtime_snode
from .utils import printd
from .dependencies import WeakDep


# Used to ensure that iterating over a set is deterministic
def tuple_sorted(x):
    if len(x) == 0:
        return []

    def sort_func(elem):
        if isinstance(elem, str):
            return elem
        elif isinstance(elem, fx.Node):
            return elem.name
        else:
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
            if isinstance(node, fx.Node):
                for inp in node.args:
                    if inp not in ancestors:
                        ancestors.add(inp)
                        new_nodes.append(inp)
            else:
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
            if isinstance(node, fx.Node):
                for inp in node.node_users:
                    if inp not in descendants:
                        descendants.add(inp)
                        new_nodes.append(inp)
            else:
                for inp in node.node_users:
                    if inp not in descendants:
                        descendants.add(inp)
                        new_nodes.append(inp)
        cur_nodes = new_nodes
    return descendants


def decide_global_ordering_comms(nodes: List["scheduler.BaseSchedulerNode"]):
    """
    Just enforces the ordering that's in the input graph.
    TODO: Come up with a better approach
    """
    comm_nodes = [n for n in nodes if isinstance(n.node, ir.CollectiveKernel)]
    for i in range(1, len(comm_nodes)):
        comm_nodes[i].add_mutation_dep(WeakDep(comm_nodes[i - 1].get_name()))


def dumb_reordering(nodes: List[fx.Node]) -> List[fx.Node]:
    """
    Sinks waits and raises comms. Does not try to reorder compute in order to
    maximize overlap.
    """
    nodes = [node for node in nodes if "fusion_meta" in node.meta]
    nodes = sink_waits(nodes)
    nodes = raise_comms(nodes)
    return [node.meta["fusion_meta"].snode for node in nodes]


def debug_print(s=""):
    if os.environ.get("INDUCTOR_COMM_DEBUG") == "1":
        printd(s)


def reorder_compute_and_comm_for_overlap(snodes: List["scheduler.BaseSchedulerNode"]) -> List[fx.Node]:
    """
    Decides a global ordering of all nodes. Assumes that we already have a global ordering of communication nodes.
    Overall strategy is:
    Priority 1. Given that we've currently scheduled comm N, we now schedule all compute nodes that are required for comm N + 1, but do not depend on comm N.
    Priority 2. Now, if all those compute nodes are sufficient to overlap comm N, we're done. Otherwise, we now need to look elsewhere to find compute that overlaps with comm N. We prioritize compute nodes that are needed sooner.
    Priority 3. Now, we schedule the compute nodes dependent on comm N and required for comm N + 1.
    Repeat.
    """

    def get_name_set(nodes_or_snodes):
        if len(nodes_or_snodes) == 0:
            return set()
        if isinstance(list(nodes_or_snodes)[0], str):
            return set(nodes_or_snodes)
        elif isinstance(list(nodes_or_snodes)[0], fx.Node):
            return set([x.name for x in nodes_or_snodes])
        else:
            return set([x.get_name() for x in nodes_or_snodes])

    def assert_equal_nodes_and_snodes(nodes, snodes):
        assert len(nodes) == len(snodes), f"nodes: {nodes} \n len(nodes): {len(nodes)} \n snodes: {snodes} \n len(snodes): {len(snodes)} \n get_name_set(snodes) ^ get_name_set(nodes): {get_name_set(snodes) ^ get_name_set(nodes)}"
        for node, snode in zip(tuple_sorted(nodes), tuple_sorted(snodes)):
            if isinstance(node, str):
                assert node == snode, f"node: {node} \n snode: {snode}"
            else:
                assert node.name == snode.get_name(), f"node.name: {node.name} \n snode.get_name(): {snode.get_name()}"

    name_to_snode = {}
    for snode in snodes:
        name_to_snode[snode.get_name()] = snode

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
    # breakpoint()
    free_nodes = set([node for node in snodes if indeg[node] == 0])

    result = []
    unused_nodes = set()
    unused_nodes = set([snode for snode in snodes])

    def add_node(snode):
        assert snode in unused_nodes
        assert snode in free_nodes
        free_nodes.remove(snode)
        unused_nodes.remove(snode)
        result.append(snode)
        for user in tuple_sorted(snode.node_users):
            if user in indeg:
                indeg[user] -= 1
                if indeg[user] == 0:
                    free_nodes.add(user)

    def add_all_nodes(snodes):
        """
        Schedules all nodes in an arbitrary topologically valid order.
        """
        all_nodes = set(snodes)
        assert all([node in unused_nodes for node in all_nodes])
        while len(all_nodes) > 0:
            # NOTE: since model graph is always a DAG and does not have circular dependency inside,
            # there should be at least one node that is a "free node" (i.e. indeg == 0),
            # hence infinite loop is not possible. But we check here just to be safe.
            progress = False
            for snode in tuple_sorted(all_nodes):
                if snode in free_nodes:
                    add_node(snode)
                    all_nodes.remove(snode)
                    progress = True
            if not progress:
                raise Exception("Unable to find a free node (indeg == 0). This is an impossible state to reach. Please report a bug to PyTorch.")

    add_all_nodes(
        list(comm_ancestors[comm_nodes[0]]) + [comm_nodes[0]],
    )

    rolled_over_compute = 0
    for idx in range(1, len(comm_ancestors)):
        is_comm_blocking = (
            len(comm_descendants[comm_nodes[idx - 1]] & comm_ancestors[comm_nodes[idx]])
            > 0
        )
        debug_print(
            f"Start {comm_nodes[idx - 1]} -> {comm_nodes[idx]} ({is_comm_blocking}, {rolled_over_compute if not is_comm_blocking else ''})"
        )
        debug_print("Priority 1")

        priority1 = unused_nodes & (
            comm_ancestors[comm_nodes[idx]] - comm_descendants[comm_nodes[idx - 1]]
        )
        total_cost = rolled_over_compute + sum(
            [get_runtime_snode(node) for node in priority1]
        )
        comm_cost = get_runtime_snode(comm_nodes[idx - 1])
        add_all_nodes(tuple_sorted(priority1))

        debug_print("Priority 2")
        group1_cost = total_cost
        if total_cost >= comm_cost:
            pass
        else:
            overlappable_nodes = tuple_sorted(
                free_nodes - comm_descendants[comm_nodes[idx - 1]]
            )

            def earliest_comm_descendant(node):
                for idx in range(len(comm_nodes)):
                    if node in comm_ancestors[comm_nodes[idx]]:
                        return idx
                return len(comm_nodes)

            overlappable_nodes = sorted(
                overlappable_nodes, key=earliest_comm_descendant
            )

            for snode in overlappable_nodes:
                if total_cost >= comm_cost:
                    break
                if not isinstance(
                    snode.node, ir.CollectiveKernel
                ):
                    runtime_cost = get_runtime_snode(snode)
                    # If we're not able to leverage more than half of this
                    # node's compute to overlap, we skip it.
                    # TODO: Smarter heuristics for packing the cost here
                    if (comm_cost - total_cost) <= runtime_cost / 2:
                        continue
                    add_node(snode)
                    total_cost += get_runtime_snode(snode)
        rollable_compute = total_cost - group1_cost
        # The idea here is that if there are no compute nodes in priority 3, we
        # can roll over the compute nodes in priority 2 to the next comm, since
        # they're not required to finish before the next comm starts

        # We can extend our ability to roll over compute if we leverage low
        # priority streams here, since that would lift us from the requirement
        # to finish priority 2 compute before the next comm starts.
        if is_comm_blocking:
            rolled_over_compute = 0
        else:
            rolled_over_compute = rollable_compute
        debug_print(f"{comm_nodes[idx-1]} overlap: {total_cost}/{comm_cost}")

        debug_print("priority 3")
        priority3 = unused_nodes & comm_ancestors[comm_nodes[idx]]
        add_all_nodes(list(priority3) + [comm_nodes[idx]])

        debug_print()

    add_all_nodes(unused_nodes)

    result = sink_waits(result)
    result = raise_comms(result)
    return result
