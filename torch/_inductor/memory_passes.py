# pyre-strict

from typing import List, Dict

import torch

from . import config, ir, scheduler
from .utils import contains_collective, contains_wait, tuple_sorted, sympy_product
import logging
from .virtualized import V

torch_log = logging.getLogger("torch")


def raise_last_usage(
    name_to_fused_node: Dict[str, "scheduler.BaseSchedulerNode"], graph_inputs: Dict[str, "Buffer"], snodes: List["scheduler.BaseSchedulerNode"]
) -> List["scheduler.BaseSchedulerNode"]:
    """
    For each node, we move its consumer nodes earlier if it satisfies the following conditions:
        - The consumer node's all input args have their write sites scheduled before or at the current node.
        - The consumer node only writes to one output tensor.
        - The consumer node's out tensor is smaller than the sum memory of all its last-usage input args.
    If we found a consumer node of current node that satisfies the above conditions, we can schedule it right after the current node.
    After moving these consumer nodes up, we are able to immediately release the memory of their last-usage input args.
    """
    last_write_map = {}  # buf -> its last write site
    for snode in snodes:
        for dep in snode.read_writes.writes:
            last_write_map[dep.name] = snode.get_name()
    for graph_input in graph_inputs.keys():
        last_write_map[graph_input] = graph_input

    def get_numel_by_name(buf_name):
        def _compute_elems(buf):
            if isinstance(buf.layout, ir.MultiOutputLayout):
                users = [user for user in name_to_fused_node[buf.get_name()].node_users if user.get_name() != buf.get_name()]
                return sum(_compute_elems(user.node) for user in users)
            else:
                return V.graph.sizevars.size_hint(sympy_product(buf.get_size()))

        if buf_name in graph_inputs:
            buf = graph_inputs[buf_name]
            return _compute_elems(buf)
        else:
            snode = name_to_fused_node[buf_name]
            if isinstance(snode, scheduler.FusedSchedulerNode):
                buf = None
                for sub_snode in snode.snodes:
                    if sub_snode.get_name() == buf_name:
                        buf = sub_snode.node
                        break
                assert buf is not None
            elif isinstance(snode, scheduler.BaseSchedulerNode):
                buf = snode.node
            return _compute_elems(buf)

    def get_users_from_unfused_nodes(snode):
        # if a fused node has 2 subnodes (A, B) and each subnode has 2 users (A1, A2) and (B1, B2),
        # this function returns (A1, A2, B1, B2).
        if isinstance(snode, scheduler._BaseGroupedSchedulerNode):
            return list(set([user for snode in snode.snodes for user in snode.users]))
        else:
            return snode.users

    new_order = []
    scheduled = set()
    scheduled.update(graph_inputs.keys())
    for snode in snodes:
        if snode.get_name() not in scheduled:
            new_order.append(snode)
            scheduled.add(snode.get_name())
            if isinstance(snode, scheduler._BaseGroupedSchedulerNode):
                for sub_snode in snode.snodes:
                    scheduled.add(sub_snode.get_name())
        for user in get_users_from_unfused_nodes(snode):
            # Can't early release `snode` if it's needed by OutputNode
            if isinstance(user.node, scheduler.OutputNode):
                break
            if contains_collective(user.node) or contains_wait(user.node):
                continue
            # For now, we don't move users that are MultiOutput
            if isinstance(user.node.node, ir.MultiOutput):
                continue
            # For now, we don't move users that are FusedSchedulerNode or GroupedSchedulerNode
            if isinstance(name_to_fused_node[user.node.get_name()], scheduler._BaseGroupedSchedulerNode):
                continue
            # For now, we don't move users that have GroupedSchedulerNode as input
            if any((not x.name in graph_inputs and isinstance(name_to_fused_node[x.name], scheduler.GroupedSchedulerNode)) for x in user.node.read_writes.reads):
                continue
            # For now, we don't move users that are run_and_save_rng_state ops
            if isinstance(user.node.node, ir.ExternKernel) and user.node.node.op_overload is torch.ops.higher_order.run_and_save_rng_state:
                continue
            if (
                all(last_write_map[x.name] in scheduled for x in user.node.read_writes.reads)
                and len(user.node.read_writes.writes) == 1 and list(user.node.read_writes.writes)[0].name == user.node.get_name()
                and get_numel_by_name(user.node.get_name()) < sum(get_numel_by_name(x_name) for x_name in user.node.last_usage)
            ):
                user_node = name_to_fused_node[user.node.get_name()]
                new_order.append(user_node)
                scheduled.add(user_node.get_name())
    return new_order
