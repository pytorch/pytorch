from typing import Any

import torch
from torch.fx import Graph, map_arg, Node
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_flatten


# flattens with support for slices
# Note: a better way to do this would
# be register/unregister slices as pytree nodes
# but there is no unregister API in the pytorch
# pytree impl
def _get_flat_args(
    node: Node, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> list[Node]:
    args = list[Any]()
    map_arg((node.args, node.kwargs), args.append)
    if node in node_to_additional_deps:
        args.extend(node_to_additional_deps[node])
    return args


def _get_flat_args_unique(
    node: Node, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> OrderedSet[Node]:
    args = OrderedSet[Node]()
    map_arg((node.args, node.kwargs), args.add)
    if node in node_to_additional_deps:
        args.update(node_to_additional_deps[node])
    return args


def _detect_cycles(
    graph: Graph, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> str:
    # States: 0=Unvisited, 1=Visiting, 2=Visited(Safe)
    state: dict[Node, int] = {}

    for root in reversed(graph.nodes):
        if root in state:
            continue

        # Stack holds (current_node, children_iterator).
        # Using an iterator allows us to pause and resume processing a node's children.
        stack = [(root, iter(_get_flat_args_unique(root, node_to_additional_deps)))]
        state[root] = 1  # Visiting

        while stack:
            parent, children = stack[-1]

            try:
                child = next(children)

                if not isinstance(child, Node):
                    continue

                child_state = state.get(child, 0)

                if child_state == 1:
                    # Back-edge: child is on the current DFS path -> cycle
                    cycle_path = [node for node, _ in stack] + [child]
                    return f"cycle detected in path: {cycle_path}"

                if child_state == 0:
                    state[child] = 1
                    stack.append(
                        (
                            child,
                            iter(_get_flat_args_unique(child, node_to_additional_deps)),
                        )
                    )
                # child_state == 2 means already verified safe; skip.

            except StopIteration:
                # All children processed — mark safe and pop.
                stack.pop()
                state[parent] = 2

    return "no cycle detected"


def _graph_device_types(graph: Graph | None) -> frozenset[str]:
    """Every device type the graph's generated code targets.

    A SCAN, not "the device of the first meta value": under dynamic shapes the
    leading placeholder is a SymInt, which has no device, so reading one value
    reported "cpu" for an all-accelerator graph -- and callers hang a toolchain
    probe and a hard load-time rejection off that answer. Values carrying no
    device (SymInt, int, None) contribute nothing rather than defaulting.

    An empty result means the graph names no device at all, which for a graph
    that emits no code is the honest answer and is not the same as "cpu".
    """
    if graph is None:
        return frozenset()

    def _device_type(x: Any) -> str | None:
        if isinstance(x, torch.device):
            return x.type
        if isinstance(x, torch.Tensor):
            return x.device.type
        return None

    def _flatten_meta(node: Node, key: str) -> list[Any]:
        if key not in node.meta:
            return []
        flat, _ = tree_flatten(node.meta[key])
        return flat

    devices: set[str] = set()
    for node in graph.nodes:
        for key in ("val", "example_value"):
            for obj in _flatten_meta(node, key):
                if (device := _device_type(obj)) is not None:
                    devices.add(device)

        # Check for device conversions
        if node.op == "call_method":
            for gpu in ["cuda", "xpu"]:
                if node.target == gpu or (node.target == "to" and gpu in node.args):
                    devices.add(gpu)

        # Check args/kwargs for non-CPU device specs
        flat_args, _ = tree_flatten((node.args, node.kwargs))
        for obj in flat_args:
            if (device := _device_type(obj)) is not None:
                devices.add(device)
    return frozenset(devices)


def _graph_device_type(graph: Graph | None) -> str:
    """The single device an artifact records for a graph.

    An accelerator wins over cpu: a mixed graph's generated code is the
    accelerator's, and "cpu" here would arm a CPU codegen-target check the
    artifact does not need.
    """
    devices = _graph_device_types(graph)
    return next((d for d in sorted(devices) if d != "cpu"), "cpu")
