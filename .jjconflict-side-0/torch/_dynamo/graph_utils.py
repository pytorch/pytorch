from typing import Any

import torch
from torch.fx import Graph, GraphModule, map_arg, Node
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


# Tensor methods whose name is the device type they move to; not every device
# type has one (there is no .mps() or .hpu()).
_DEVICE_NAMING_METHODS = ("cpu", "cuda", "xpu", "ipu", "mtia")


def _graph_device_types(
    graph: Graph | None, _seen: set[Graph] | None = None
) -> frozenset[str]:
    """Every device type the graph names -- from a meta value, a device-naming
    method or a device position (a device= kwarg, .to()'s device argument), in
    this graph and in the submodule bodies it reaches through a get_attr --
    except "meta", which is abstract rather than a requirement of the host. An
    empty result means the graph names no device, which is not "cpu". A body
    only set as an attribute, as the saved tensors hooks subgraphs are, is not
    reached.
    """
    if graph is None:
        return frozenset()
    # A reused body has one get_attr per call site, so scanning it per node is
    # exponential in nesting depth, and one reachable from itself never ends.
    if _seen is None:
        _seen = set()
    if graph in _seen:
        return frozenset()
    _seen.add(graph)

    def _device_type(x: object) -> str | None:
        if isinstance(x, torch.device):
            return x.type
        if isinstance(x, torch.Tensor):
            return x.device.type
        return None

    def _device_from_spec(x: object) -> str | None:
        # A bare string or index in a device position names a device, and
        # Dynamo emits both (device=0, x.to(0)); a value torch.device rejects
        # names no device rather than aborting the compile, and bool is kept out
        # of the index arm because torch.device rejects it with a TypeError this
        # does not catch. An index carries no device type of its own:
        # deviceFromLong resolves it through at::getAccelerator(true), the
        # accelerator of the process doing the compile and never the target's,
        # so a device=0 in a graph traced for another target contributes this
        # build's accelerator or nothing.
        if isinstance(x, str) or (isinstance(x, int) and not isinstance(x, bool)):
            try:
                return torch.device(x).type
            except (RuntimeError, ValueError):
                return None
        return _device_type(x)

    def _flatten_meta(node: Node, key: str) -> list[object]:
        if key not in node.meta:
            return []
        flat, _ = tree_flatten(node.meta[key])
        return flat

    def _device_specs(node: Node) -> list[object]:
        # The only positions this scan reads as devices: a device anywhere else
        # (an autocast string, aten.to.device's positional Device) is not read
        # as one, though such a node's meta names the device it returns. Not
        # every real device position is here: x.type() takes a
        # "torch.cuda.FloatTensor", which torch.device does not parse.
        specs: list[object] = []
        if "device" in node.kwargs:
            specs.append(node.kwargs["device"])
        if node.op == "call_method" and node.target == "to" and len(node.args) >= 2:
            # args[1] is a device only in some overloads (x.to(torch.float16)
            # lands here too, and names no device).
            specs.append(node.args[1])
        return specs

    # The rename can happen after this module is imported, so read it here.
    naming_methods = (*_DEVICE_NAMING_METHODS, torch._C._get_privateuse1_backend_name())
    devices: set[str] = set()
    for node in graph.nodes:
        for key in ("val", "example_value"):
            for obj in _flatten_meta(node, key):
                if (device := _device_type(obj)) is not None:
                    devices.add(device)

        # x.cuda() names the device in the method, not in an argument.
        if node.op == "call_method" and node.target in naming_methods:
            devices.add(node.target)

        for obj in _device_specs(node):
            if (device := _device_from_spec(obj)) is not None:
                devices.add(device)

        # A HOP body (a cond branch, an invoke_subgraph region) is a submodule
        # this graph only references, and the parent node's meta shows what the
        # body returned rather than the devices it used. A get_attr target is a
        # qualified name, so resolve it one atom at a time as FX does.
        if node.op == "get_attr" and (owner := graph.owning_module) is not None:
            sub: object = owner
            for atom in node.target.split("."):
                sub = getattr(sub, atom, None)
            if isinstance(sub, GraphModule):
                devices |= _graph_device_types(sub.graph, _seen)
    return frozenset(devices) - {"meta"}
