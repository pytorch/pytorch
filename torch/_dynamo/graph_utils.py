from collections import deque
from typing import Any

from torch.fx import Graph, Node
from torch.utils._pytree import tree_flatten


# flattens with support for slices
# Note: a better way to do this would
# be register/unregister slices as pytree nodes
# but there is no unregister API in the pytorch
# pytree impl
def _flatten_args_kwargs(args: Any) -> list[Node]:
    fully_flattened = []

    def flatten(args: Any) -> None:
        flattened, _ = tree_flatten(args)
        for arg in flattened:
            if isinstance(arg, slice):
                start = arg.start
                stop = arg.stop
                step = arg.step
                flatten((start, stop, step))
            else:
                fully_flattened.append(arg)

    flatten(args)

    return fully_flattened


def _detect_cycles(graph: Graph) -> str:
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

    for origin in graph.find_nodes(op="placeholder"):
        current_path.clear()
        current_path_set.clear()
        add_to_current_path(origin)
        for child in origin.users:
            pending.append((child, origin))

        while pending:
            cur_node, parent = pending.pop()

            while current_path_head() != parent:
                pop_current_path()

            if cur_node in current_path_set:
                current_path.append(cur_node)
                return f"cycle detected in path: {current_path}"

            add_to_current_path(cur_node)
            for child in cur_node.users:
                pending.append((child, cur_node))

    return "no cycle detected"
