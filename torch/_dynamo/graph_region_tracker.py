from collections import defaultdict, deque
from typing import Dict, List, Set

import torch.fx


Region = Set[torch.fx.Node]
IdenticalNodes = Set[torch.fx.Node]


# This is typical BFS with the caveat
# that a node's children need to be explicitly
# added with the add_children() method
# The flow is yield a node and check if it's valid for all regions
# if not valid, discard and continue onto the next node
class BfsRegionIter:
    def __init__(self, origin):
        self._cur_node = (None, origin)
        self._queue = deque()

    @staticmethod
    def create(origin):
        it = BfsRegionIter(origin)
        it.add_children(origin)
        return it

    def next(self):
        ret_node = self._cur_node
        if not self._queue:
            self._cur_node = (None, None)
        else:
            self._cur_node = self._queue.popleft()
        return ret_node

    def peek(self):
        return self._cur_node

    def add_children(self, node):
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                self._queue.append((None, arg))

        for key, kwarg in node.kwargs:
            if isinstance(kwarg, torch.fx.Node):
                self._queue.append((key, kwarg))


class GraphRegionTracker:
    def __init__(self):
        self.loc_to_duplicates: Dict[str, IdenticalNodes] = defaultdict(set)
        self.node_to_duplicates: Dict[torch.fx.Node, IdenticalNodes] = {}

    @staticmethod
    def _get_loc_str(filename: str, lineno: int):
        return f"{filename}:{lineno}"

    def track_node(self, filename: str, lineno: int, node: torch.fx.Node):
        loc_str = self._get_loc_str(filename, lineno)
        duplicates = self.loc_to_duplicates[loc_str]
        duplicates.add(node)
        self.node_to_duplicates[node] = duplicates

    def has_same_loc(self, n0, n1):
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

    def fully_expand_region_group(self, regions: List[Region]):
        # All regions should start with 1 node
        assert all(len(region) == 1 for region in regions)
        region_iters = []
        for region in regions:
            (origin,) = region  # Only works for 1 element sets
            region_iters.append(BfsRegionIter.create(origin))

        nodes_to_add = []
        seen_nodes = set()

        # arg_name is set for kwargs, None for args
        current_arg_name, current_node = region_iters[0].next()
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
                else:
                    add_node = False

                seen_nodes.add(node)

            if add_node:
                for region, region_it, node in zip(regions, region_iters, nodes_to_add):
                    region.add(node)
                    region_it.add_children(node)

            current_arg_name, current_node = region_iters[0].next()

    def __str__(self):
        return f"GraphRegionTracker(loc_to_duplicates={self.loc_to_duplicates}, node_to_duplicates={self.node_to_duplicates})"
