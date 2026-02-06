from collections import deque
from typing import Any, Optional

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
    
    # We iterate over all output nodes to cover the graph backwards
    all_nodes = reversed(graph.nodes)
    
    for root in all_nodes:
        if root in state:
            continue
            
        # Stack holds tuples of: (current_node, children_iterator)
        # Using an iterator allows us to pause and resume processing a node
        stack = [(root, iter(_get_flat_args_unique(root, node_to_additional_deps)))]
        state[root] = 1  # Mark as Visiting
        
        while stack:
            parent, children = stack[-1]
            
            try:
                # Get the next child to visit
                child = next(children)
                
                if not isinstance(child, Node):
                    continue
                
                child_state = state.get(child, 0)
                
                if child_state == 1:
                    # Found a node that is currently being visited -> CYCLE DETECTED!
                    cycle_path = [node for node, _ in stack] + [child]
                    return f"cycle detected in path: {cycle_path}"
                
                if child_state == 0:
                    # Found an unvisited node -> Push to stack and mark as Visiting
                    state[child] = 1
                    stack.append((child, iter(_get_flat_args_unique(child, node_to_additional_deps))))
                
                # If child_state == 2:
                # It means this node and its subgraph are already checked and safe.
                # We do nothing and continue to the next child.
                
            except StopIteration:
                # All children of 'parent' have been processed.
                # Mark 'parent' as Visited (Safe) and pop from stack.
                stack.pop()
                state[parent] = 2
                
    return "no cycle detected"


def _graph_device_type(graph: Optional[Graph]) -> str:
    if graph is None:
        return "cpu"

    def _device_type(x: Any) -> str:
        if isinstance(x, torch.device):
            return x.type
        if isinstance(x, torch.Tensor):
            return x.device.type
        return "cpu"

    def _flatten_meta(node: Node, key: str) -> list[Any]:
        if key not in node.meta:
            return []
        flat, _ = tree_flatten(node.meta[key])
        return flat

    for node in graph.nodes:
        for key in ("val", "example_value"):
            for obj in _flatten_meta(node, key):
                return _device_type(obj)

        # Check for device conversions
        if node.op == "call_method":
            for gpu in ["cuda", "xpu"]:
                if node.target == gpu:
                    return gpu
                if node.target == "to" and gpu in node.args:
                    return gpu

        # Check args/kwargs for non-CPU device specs
        flat_args, _ = tree_flatten((node.args, node.kwargs))
        for obj in flat_args:
            return _device_type(obj)
    return "cpu"
