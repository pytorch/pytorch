import collections
from typing import Any, List

import torch
import torch.fx as fx
from . import ir


def create_fx_from_snodes(snodes: List["scheduler.BaseSchedulerNode"]) -> fx.Graph:
    """
    Creates a FX Graph from a list of SchedulerNode objects.
    """
    from . import ir, scheduler

    def get_fake_func(name):
        def func1(*args):
            return 0

        func1.__name__ = name
        return func1

    FusionMeta = collections.namedtuple("FusionMeta", ["group", "snode", "type"])

    func_dict = {
        s: get_fake_func(s) for s in ["extern", "nop", "compute", "fused", "collective"]
    }
    buf_to_fx_node = {}
    graph = torch.fx.Graph()
    first_node = None

    outputs = []
    group: Any = None
    # create call_function node for each Buffer and Kernel
    for snode in snodes:
        if isinstance(snode.node, (ir.CollectiveKernel, ir.Wait)):
            node_type = "collective"
            group = node_type
        elif snode.is_extern():
            node_type = "extern"
            group = node_type
        elif snode.is_template():
            node_type = "template"
            group = node_type
        elif isinstance(snode, scheduler.NopKernelSchedulerNode):
            node_type = "nop"
            group = node_type
        elif isinstance(snode, scheduler.SchedulerNode):
            node_type = "compute"
            group = snode.group
        elif isinstance(snode, scheduler.FusedSchedulerNode):
            node_type = "fused"
            group = snode.group
        else:
            raise RuntimeError("Unknown node type")
        node_func = func_dict[node_type]
        fx_node = graph.call_function(node_func, args=(), kwargs=None)

        def in_output(snode):
            if isinstance(snode, scheduler.FusedSchedulerNode):
                return any(in_output(x) for x in snode.snodes)
            return any(
                isinstance(user.node, scheduler.OutputNode) for user in snode.users
            )

        if in_output(snode):
            outputs.append(fx_node)
        name = snode.get_name()
        fx_node.name = name

        fx_node.meta["fusion_meta"] = FusionMeta(group, snode, node_type)

        if isinstance(snode, scheduler.FusedSchedulerNode):
            for x in snode.snodes:
                buf_to_fx_node[x.get_name()] = fx_node
        buf_to_fx_node[name] = fx_node

        if first_node is None:
            first_node = fx_node

    # create edges between nodes
    for snode in snodes:
        name = snode.get_name()
        deps = snode.unmet_dependencies

        fx_node = buf_to_fx_node[name]
        new_args = []
        for dep in deps:
            if dep.name in buf_to_fx_node:
                dep_node = buf_to_fx_node[dep.name]
            else:
                with graph.inserting_before(first_node):
                    dep_node = graph.placeholder(dep.name)
                    buf_to_fx_node[dep.name] = dep_node
            new_args.append(dep_node)

        fx_node.args = tuple(new_args)

    graph.output(outputs[0] if len(outputs) == 1 else tuple(outputs))
    return graph


def get_runtime_snode(snode: "BaseSchedulerNode"):
    """
    Gets the runtime of Scheduler node. Currently somewhat of a placeholder, todo to be replaced by more sophisticated approaches.
    """
    if isinstance(snode.node, ir.AllReduce):
        return 5
    if isinstance(snode.node, ir.CollectiveKernel):
        return 15
    if isinstance(snode.node, ir.MultiOutput):
        return 0
    if isinstance(snode.node, ir.Wait):
        return 0
    if isinstance(snode.node, ir.ExternKernel):
        return 10
    return 1
