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


# Tensor methods whose name is the device type they move to. Not every device
# type has one (there is no .mps() or .hpu()), and a renamed PrivateUse1
# backend generates one this tuple does not carry (.npu(), for a backend
# renamed npu). x.cpu() names cpu as explicitly as x.cuda() names cuda, so it
# belongs here even though the collapse reads an empty answer as cpu anyway.
_DEVICE_NAMING_METHODS = ("cpu", "cuda", "xpu", "ipu", "mtia")


def _graph_device_types(graph: Graph | None) -> frozenset[str]:
    """Every device type named by the graph's meta values, by a device-naming
    method (_DEVICE_NAMING_METHODS) or by a device position (a device= kwarg,
    .to()'s device argument), except "meta", which is an abstract device rather
    than a runtime requirement of the host. Values that name no device (a
    SymInt, None, a dtype) contribute nothing; an empty result means the graph
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
        # -- so a bare string or a bare index names a device here (Dynamo emits
        # both: device=0 and x.to(0) reach this as an int, which torch.device
        # resolves against the accelerator the BUILD provides, a registered
        # PrivateUse1 backend first, not one this host can currently use).
        # Autocast device types (_enter_autocast('cuda', ...)) are ordinary
        # positional args, never a device position, so they cannot reach this
        # and inject a device no tensor lives on. Not every value parses, so
        # let torch.device reject: an unknown device name raises RuntimeError,
        # an index that does not fit int64 raises ValueError.
        if isinstance(x, str) or (isinstance(x, int) and not isinstance(x, bool)):
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
        # The device positions this scan reads. Most Dynamo nodes name the
        # device in their meta already (`output` and the autocast enter/exit
        # markers do not), so this is what a graph WITHOUT meta is read by. Not
        # every position that names a device is here: x.type() takes a bare
        # "torch.cuda.FloatTensor", which names cuda in a string torch.device
        # does not parse, so a meta-less graph whose only signal is that reads
        # as no device.
        specs: list[Any] = []
        if "device" in node.kwargs:
            specs.append(node.kwargs["device"])
        if node.op == "call_method" and node.target == "to" and len(node.args) >= 2:
            # args[0] is the tensor; args[1] is .to()'s first argument, which is
            # a device only in some overloads (x.to(torch.float16) lands here
            # too, and names no device).
            specs.append(node.args[1])
        return specs

    devices: set[str] = set()
    for node in graph.nodes:
        for key in ("val", "example_value"):
            for obj in _flatten_meta(node, key):
                if (device := _device_type(obj)) is not None:
                    devices.add(device)

        # x.cuda() and friends name the device in the method itself, so there
        # is no value in a device position to read.
        if node.op == "call_method" and node.target in _DEVICE_NAMING_METHODS:
            devices.add(node.target)

        for obj in _device_specs(node):
            if (device := _device_from_spec(obj)) is not None:
                devices.add(device)
    # meta is an abstract device, never a runtime requirement of the host.
    return frozenset(devices) - {"meta"}


def _collapse_device_types(device_types: frozenset[str]) -> str:
    """The single device type both callers record: an accelerator wins over cpu,
    and naming no device at all reads as cpu, because such a graph lowers to CPU
    code. Among several accelerators the pick is arbitrary (alphabetical).
    """
    return next((d for d in sorted(device_types) if d != "cpu"), "cpu")
