from collections import deque
from typing import Any

from torch.fx import Graph, map_arg, Node
from torch.utils._ordered_set import OrderedSet


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
    current_path: deque[Node] = deque()
    current_path_set: set[Node] = set()
    pending: deque[tuple[Node, Node]] = deque()

    def add_to_current_path(node: Node) -> None:
        current_path.append(node)
        current_path_set.add(node)

    def pop_current_path() -> None:
        node = current_path.pop()
        current_path_set.remove(node)

    def current_path_head() -> Node:
        return current_path[-1]

    for origin in graph.find_nodes(op="output"):
        current_path.clear()
        current_path_set.clear()
        add_to_current_path(origin)
        for child in _get_flat_args_unique(origin, node_to_additional_deps):
            pending.append((child, origin))

        while pending:
            cur_node, parent = pending.pop()

            # handle backtracking
            while current_path and current_path_head() != parent:
                pop_current_path()

            if not isinstance(cur_node, Node):
                continue

            if cur_node in current_path_set:
                current_path.append(cur_node)
                return f"cycle detected in path: {current_path}"

            add_to_current_path(cur_node)

            for child in _get_flat_args_unique(cur_node, node_to_additional_deps):
                pending.append((child, cur_node))

    return "no cycle detected"
