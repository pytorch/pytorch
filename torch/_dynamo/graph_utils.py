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
    """Every device type named by the graph's meta values or device positions
    (a device= kwarg, a .to()/.cuda()/.xpu() call). Values with no device
    (SymInt, int, None) contribute nothing; an empty result means the graph
    names no device, which is not "cpu".
    """
    if graph is None:
        return frozenset()

    def _device_type(x: Any) -> str | None:
        if isinstance(x, torch.device):
            return x.type
        if isinstance(x, torch.Tensor):
            return x.device.type
        return None

    def _device_from_spec(x: Any) -> str | None:
        # x sits in a device position -- a device= kwarg or .to()'s device arg
        # -- so a bare string here names a device. Autocast device types
        # (_enter_autocast('cuda', ...)) are ordinary positional args, never a
        # device position, so they cannot reach this and inject a device no
        # tensor lives on. Not every string parses, so let torch.device reject.
        if isinstance(x, str):
            try:
                return torch.device(x).type
            except (RuntimeError, ValueError):
                return None
        return _device_type(x)

    def _flatten_meta(node: Node, key: str) -> list[Any]:
        if key not in node.meta:
            return []
        flat, _ = tree_flatten(node.meta[key])
        return flat

    def _device_specs(node: Node) -> list[Any]:
        # The only node positions where a bare string names the graph's device.
        specs: list[Any] = []
        if "device" in node.kwargs:
            specs.append(node.kwargs["device"])
        if node.op == "call_method" and node.target == "to" and len(node.args) >= 2:
            specs.append(node.args[1])  # args[0] is the tensor; args[1] its target
        return specs

    devices: set[str] = set()
    for node in graph.nodes:
        for key in ("val", "example_value"):
            for obj in _flatten_meta(node, key):
                if (device := _device_type(obj)) is not None:
                    devices.add(device)

        # x.cuda() / x.xpu() name the device in the method itself, with no
        # device leaf to read.
        if node.op == "call_method" and node.target in ("cuda", "xpu"):
            devices.add(node.target)

        for obj in _device_specs(node):
            if (device := _device_from_spec(obj)) is not None:
                devices.add(device)
    # meta is an abstract device, never a runtime requirement of the host.
    return frozenset(devices) - {"meta"}
