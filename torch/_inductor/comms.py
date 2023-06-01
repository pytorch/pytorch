from collections import defaultdict
from typing import List

import torch.fx as fx
from . import ir
from .analysis import get_runtime_snode
from .utils import is_local, print2


# Used to ensure that iterating over a set is deterministic
def tuple_sorted(x):
    return sorted(tuple(x), key=lambda x: x.name)


def sink_waits(result: List[fx.Node]) -> List[fx.Node]:
    new_result = []
    cur_waits = set()
    for node in result:
        if isinstance(node.meta["fusion_meta"].snode.node, ir.Wait):
            cur_waits.add(node)
        else:
            for wait in tuple_sorted(cur_waits):
                if node in wait.users:
                    new_result.append(wait)
                    cur_waits.remove(wait)

            new_result.append(node)
    for node in tuple_sorted(cur_waits):
        new_result.append(node)
    return new_result


def raise_comms(result: List[fx.Node]) -> List[fx.Node]:
    new_result = []
    cur_comms = []
    for node in reversed(result):
        if isinstance(node.meta["fusion_meta"].snode.node, ir.CollectiveKernel):
            cur_comms.append(node)
        else:
            while len(cur_comms) > 0 and any([node in comm.args for comm in cur_comms]):
                comm = cur_comms.pop(0)
                new_result.append(comm)
            new_result.append(node)
    assert len(cur_comms) <= 1
    for node in tuple_sorted(cur_comms):
        new_result.append(node)
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
            for inp in node.users:
                if inp not in descendants:
                    descendants.add(inp)
                    new_nodes.append(inp)
        cur_nodes = new_nodes
    return descendants


def order_heuristic(nodes: List[fx.Node]) -> List[fx.Node]:
    comm_nodes = []
    for node in nodes:
        if "fusion_meta" in node.meta and isinstance(
            node.meta["fusion_meta"].snode.node, ir.CollectiveKernel
        ):
            comm_nodes.append(node)

    new_nodes = [node for node in nodes if "fusion_meta" in node.meta]
    if len(comm_nodes) == 0:
        return new_nodes

    comm_ancestors = {node: get_ancestors(node) for node in comm_nodes}
    comm_descendants = {node: get_descendants(node) for node in comm_nodes}

    # new_nodes = sink_waits(new_nodes)
    # new_nodes = raise_comms(new_nodes)
    # return [node.meta['fusion_meta'].snode for node in new_nodes]
    indeg = {k: 0 for k in new_nodes}
    buf_uses = defaultdict(set)
    for node in new_nodes:
        snode = node.meta["fusion_meta"].snode
        for buf in snode.used_buffer_names():
            buf_uses[buf].add(snode)
        for user in node.users:
            if user in indeg:
                indeg[user] += 1
    free_nodes = set([node for node in new_nodes if indeg[node] == 0])

    result = []
    unused_nodes = set([node for node in nodes if "fusion_meta" in node.meta])

    def add_node(node):
        assert node in unused_nodes
        assert node in free_nodes
        print2(f"adding {node}")
        free_nodes.remove(node)
        unused_nodes.remove(node)
        result.append(node)
        for user in node.users:
            if user in indeg:
                indeg[user] -= 1
                if indeg[user] == 0:
                    free_nodes.add(user)

    def add_all_nodes(nodes):
        """
        Schedules all nodes in an arbitrary topologically valid order.
        """
        all_nodes = set(nodes)
        assert all([node in unused_nodes for node in all_nodes])
        while len(all_nodes) > 0:
            for node in tuple_sorted(all_nodes):
                if node in free_nodes:
                    add_node(node)
                    all_nodes.remove(node)

    add_all_nodes(list(comm_ancestors[comm_nodes[0]]) + [comm_nodes[0]])

    def get_runtime_fx(node):
        return get_runtime_snode(node.meta["fusion_meta"].snode)

    for idx in range(1, len(comm_ancestors)):
        is_comm_blocking = (
            len(comm_descendants[comm_nodes[idx - 1]] & comm_ancestors[comm_nodes[idx]])
            > 0
        )
        print2(f"Start {comm_nodes[idx - 1]} -> {comm_nodes[idx]} ({is_comm_blocking})")
        priority1 = unused_nodes & (
            comm_ancestors[comm_nodes[idx]] - comm_descendants[comm_nodes[idx - 1]]
        )
        total_cost = sum([get_runtime_fx(node) for node in priority1])
        comm_cost = get_runtime_fx(comm_nodes[idx - 1])
        print2("Priority 1")
        add_all_nodes(tuple_sorted(priority1))
        print2("Priority 2")

        if total_cost > comm_cost:
            pass
        else:
            while total_cost < comm_cost:
                compute_overlap_node = None
                for node in tuple_sorted(
                    free_nodes - comm_descendants[comm_nodes[idx - 1]]
                ):
                    if not isinstance(
                        node.meta["fusion_meta"].snode.node, ir.CollectiveKernel
                    ):
                        compute_overlap_node = node
                        break
                if compute_overlap_node is None:
                    break

                add_node(compute_overlap_node)
                total_cost += get_runtime_fx(compute_overlap_node)
        print2(f"{comm_nodes[idx-1]} overlap: {total_cost}/{comm_cost}")
        # if is_local():
        #     print(f"{comm_nodes[idx - 1]} -> {comm_nodes[idx]}", total_cost, comm_cost)
        print2("priority 3")
        priority3 = unused_nodes & comm_ancestors[comm_nodes[idx]]
        add_all_nodes(list(priority3) + [comm_nodes[idx]])
        print2()

    add_all_nodes(unused_nodes)

    result = sink_waits(result)
    result = raise_comms(result)

    print2(result)
    return result
