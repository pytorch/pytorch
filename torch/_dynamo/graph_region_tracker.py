from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import torch.fx


Node = torch.fx.Node
Region = Set[Node]
IdenticalNodes = Set[Node]


# This is typical BFS with the caveat
# that a node's children need to be explicitly
# added with the add_children() method
# The flow is yield a node and check if it's valid for all regions
# if not valid, discard and continue onto the next node
class BfsRegionIter:
    def __init__(self, origin: Node) -> None:
        self._cur_node: Tuple[Optional[str], Optional[Node]] = (None, origin)
        self._queue: Deque[Tuple[Optional[str], Optional[Node]]] = deque()

    @staticmethod
    def create(origin: Node) -> "BfsRegionIter":
        it = BfsRegionIter(origin)
        it.add_children(origin)
        return it

    def next(self) -> Tuple[Optional[str], Optional[Node]]:
        ret_node = self._cur_node
        if not self._queue:
            self._cur_node = (None, None)
        else:
            self._cur_node = self._queue.popleft()
        return ret_node

    def peek(self) -> Tuple[Optional[str], Optional[Node]]:
        return self._cur_node

    def add_children(self, node: Node) -> None:
        arg: Any
        for arg in node.args:
            if isinstance(arg, Node):
                self._queue.append((None, arg))

        key: str
        kwarg: Any
        for key, kwarg in node.kwargs.items():
            if isinstance(kwarg, Node):
                self._queue.append((key, kwarg))


class GraphRegionTracker:
    def __init__(self) -> None:
        self.loc_to_duplicates: Dict[str, IdenticalNodes] = defaultdict(set)
        self.node_to_duplicates: Dict[Node, IdenticalNodes] = {}

    @staticmethod
    def _get_loc_str(filename: str, lineno: int) -> str:
        return f"{filename}:{lineno}"

    def track_node(self, filename: str, lineno: int, node: Node) -> None:
        loc_str = self._get_loc_str(filename, lineno)
        duplicates = self.loc_to_duplicates[loc_str]
        duplicates.add(node)
        self.node_to_duplicates[node] = duplicates

    def has_same_loc(self, n0: Node, n1: Node) -> bool:
        return (
            n0 in self.node_to_duplicates
            and n1 in self.node_to_duplicates
            and self.node_to_duplicates[n0] == self.node_to_duplicates[n1]
        )

    def get_identical_regions(self) -> List[List[Region]]:
        region_groups = [
            [{n} for n in group]
            for group in self.loc_to_duplicates.values()
            if len(group) > 1
        ]

        for region_group in region_groups:
            self.fully_expand_region_group(region_group)

        return region_groups

    def fully_expand_region_group(self, regions: List[Region]) -> None:
        # All regions should start with 1 node
        assert all(len(region) == 1 for region in regions)
        region_iters = []
        for region in regions:
            (origin,) = region  # Only works for 1 element sets
            region_iters.append(BfsRegionIter.create(origin))

        nodes_to_add: List[Node] = []
        seen_nodes: Set[Node] = set()

        # arg_name is set for kwargs, None for args
        current_arg_name, current_node = region_iters[0].next()
        assert current_node is not None
        seen_nodes.add(current_node)
        while current_node:
            add_node = True
            nodes_to_add.clear()
            nodes_to_add.append(current_node)
            for region_it in region_iters[1:]:
                arg_name, node = region_it.next()

                if node:
                    add_node &= (
                        current_arg_name == arg_name
                        and node not in seen_nodes
                        and self.has_same_loc(node, current_node)
                    )
                    nodes_to_add.append(node)
                    seen_nodes.add(node)
                else:
                    add_node = False

            if add_node:
                for region, region_it, node in zip(regions, region_iters, nodes_to_add):
                    region.add(node)
                    region_it.add_children(node)

            current_arg_name, current_node = region_iters[0].next()

    def __str__(self) -> str:
        return f"GraphRegionTracker(loc_to_duplicates={self.loc_to_duplicates}, node_to_duplicates={self.node_to_duplicates})"
