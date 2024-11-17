import itertools
import math
import operator
from collections import defaultdict, deque
from typing import Any, Callable, DefaultDict, Deque, Dict, List, Optional, Set, Tuple

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

    def get_identical_regions(self, gm: torch.fx.GraphModule) -> List[List[Region]]:
        topological_ranking = {node: i for i, node in enumerate(gm.graph.nodes)}
        group_ranking = {}
        region_groups = []

        # Create region groups; a region group is a group
        # of regions that are all identical. In this initial state
        # each region in the group is a single node, and we discard
        # groups that are only a single region.
        # We track the topological ranking to start with groups later in the graph
        # the reason for this is that we will necessarily create the largest groups first.
        for group in self.loc_to_duplicates.values():
            if len(group) > 1:
                region_group = []
                min_rank = math.inf
                for node in group:
                    min_rank = min(min_rank, topological_ranking[node])
                    region_group.append({node})

                region_groups.append(region_group)
                group_ranking[id(region_group)] = min_rank

        region_groups.sort(key=lambda g: -group_ranking[id(g)])

        # We start from regions later in the graph and expand them earlier
        # as a result, we will create the largest regions first and they won't
        # overlap.
        seen_nodes: Set[Node] = set()
        for region_group in region_groups:
            fully_expand_region_group(region_group, seen_nodes, self.has_same_loc)

        region = region_group[0]
        graph = create_graph_from_region(region)

        return region_groups

    def __str__(self) -> str:
        return f"GraphRegionTracker(loc_to_duplicates={self.loc_to_duplicates}, node_to_duplicates={self.node_to_duplicates})"


def fully_expand_region_group(
    regions: List[Region],
    seen_nodes: Set[Node],
    has_same_loc_fn: Callable[[Node, Node], bool],
) -> None:
    # All regions should start with 1 node
    assert all(len(region) == 1 for region in regions)
    region_iters = []
    for region in regions:
        (origin,) = region  # Only works for 1 element sets
        region_iters.append(BfsRegionIter.create(origin))

    nodes_to_add: List[Node] = []

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
                    and has_same_loc_fn(node, current_node)
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


def apply_graph_deduplication(output_graph) -> None:  # type: ignore[no-untyped-def]
    duplicated_region_groups = output_graph.graph_region_tracker.get_identical_regions(
        output_graph.nn_modules
    )
    for region_group in duplicated_region_groups:
        region = region_group[0]
        subgraph, node_to_subgraph_output = create_graph_from_region(region)
        sub_gm = torch.fx.GraphModule(output_graph.nn_modules, subgraph)
        subgraph_name = output_graph.install_subgraph("subgraph", sub_gm)
        subgraph_node = output_graph.create_proxy("get_attr", subgraph_name, (), {})
        for region in region_group:
            replace_region_with_subgraph(
                output_graph.graph,
                region,
                subgraph_node,
                node_to_subgraph_output,
                subgraph_name,
                subgraph,
            )


def replace_region_with_subgraph(
    graph: torch.fx.Graph,
    region: Region,
    subgraph_node: Node,
    node_to_subgraph_output: Dict[Node, Tuple[int, Node]],
    subgraph_name: str,
    subgraph: torch.fx.Graph,
) -> None:
    # need to fill in inputs properly here
    invoke_subgraph_node = graph.create_node(
        "call_function", torch.ops.higher_order.invoke_subgraph, (subgraph_node,), {}
    )
    for node, (ind, _) in node_to_subgraph_output.items():
        subgraph_output = graph.create_node(
            "call_function", operator.getitem, (invoke_subgraph_node, ind), {}
        )
        node.replace_all_uses_with(subgraph_output)

    for node in region:
        graph.erase_node(node)


def get_node_to_external_inputs(
    region: Region,
) -> Tuple[DefaultDict[Node, List[Node]], Set[Node]]:
    node_to_inputs = defaultdict(list)
    external_nodes = set()
    for node in region:
        for in_node in itertools.chain(node.args, node.kwargs.values()):
            if in_node not in region:
                node_to_inputs[node].append(in_node)
                external_nodes.add(in_node)

    return node_to_inputs, external_nodes  # type: ignore[return-value]


def get_node_to_external_users(
    region: Region,
) -> Tuple[DefaultDict[Node, List[Node]], Set[Node]]:
    outputs_to_users = defaultdict(list)
    external_nodes = set()
    for node in region:
        for user in node.users:
            if user not in region:
                outputs_to_users[node].append(user)
                external_nodes.add(user)

    return outputs_to_users, external_nodes


def copy_nodes_and_remap_inputs(
    subgraph: torch.fx.Graph, region: Region
) -> Dict[Node, Node]:
    node_to_inputs, external_inputs = get_node_to_external_inputs(region)
    arg_remap = {}
    for node in external_inputs:
        arg_remap[node] = subgraph.placeholder(f"subgraph_input_{node.name}")

    def map_arg(node: Node) -> Node:
        if node in arg_remap:
            return arg_remap[node]
        else:
            return node

    old_to_new_node = {}
    for node in region:
        old_to_new_node[node] = subgraph.node_copy(node, lambda old: map_arg(old))

    return old_to_new_node


def create_outputs(
    subgraph: torch.fx.Graph, old_to_new_nodes: Dict[Node, Node], region: Region
) -> Dict[Node, Tuple[int, Node]]:
    outputs_to_users, external_users = get_node_to_external_users(region)
    output_node_to_subgraph_output = {}

    for ind, (output_node, _) in enumerate(outputs_to_users.items()):
        output_node_to_subgraph_output[output_node] = (
            ind,
            subgraph.output(old_to_new_nodes[output_node]),
        )

    return output_node_to_subgraph_output


def create_graph_from_region(
    region: Region,
) -> Tuple[torch.fx.Graph, Dict[Node, Tuple[int, Node]]]:
    subgraph: torch.fx.Graph = torch.fx.Graph()
    old_to_new_nodes = copy_nodes_and_remap_inputs(subgraph, region)
    output_node_to_subgraph_output = create_outputs(subgraph, old_to_new_nodes, region)
    return subgraph, output_node_to_subgraph_output
