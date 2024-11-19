import math
import operator
from collections import defaultdict, deque
from typing import (
    Any,
    Callable,
    DefaultDict,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

import torch.fx
from torch.utils._pytree import tree_flatten


Node = torch.fx.Node
Region = List[Node]
IdenticalNodes = List[Node]


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
        self.loc_to_duplicates: Dict[str, IdenticalNodes] = defaultdict(list)
        self.node_to_duplicates: Dict[Node, IdenticalNodes] = {}

    @staticmethod
    def _get_loc_str(filename: str, lineno: int) -> str:
        return f"{filename}:{lineno}"

    def track_node(self, filename: str, lineno: int, node: Node) -> None:
        loc_str = self._get_loc_str(filename, lineno)
        duplicates = self.loc_to_duplicates[loc_str]
        duplicates.append(node)
        self.node_to_duplicates[node] = duplicates

    def has_same_loc(self, n0: Node, n1: Node) -> bool:
        return (
            n0 in self.node_to_duplicates
            and n1 in self.node_to_duplicates
            and self.node_to_duplicates[n0] == self.node_to_duplicates[n1]
        )

    def get_identical_regions(self, graph: torch.fx.Graph) -> List[List[Region]]:
        topological_ranking = {node: i for i, node in enumerate(graph.nodes)}
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
                    region_group.append([node])

                region_groups.append(region_group)
                group_ranking[id(region_group)] = min_rank

        region_groups.sort(key=lambda g: -group_ranking[id(g)])

        # We start from regions later in the graph and expand them earlier
        # as a result, we will create the largest regions first and they won't
        # overlap.
        seen_nodes: Set[Node] = set()
        for region_group in region_groups:
            fully_expand_region_group(region_group, seen_nodes, self.has_same_loc)

        return [
            region_group for region_group in region_groups if len(region_group[0]) > 1
        ]

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

    # we already have the origin node in each region
    for region_it in region_iters:
        _, node = region_it.next()
        assert node
        region_it.add_children(node)

    # arg_name is set for kwargs, None for args
    current_arg_name, current_node = region_iters[0].next()
    assert current_node is not None
    seen_nodes.add(current_node)
    # Loop incrementally adding new nodes to each region
    # regions are only expanded if the node to add is valid
    # for ALL regions
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
                region.append(node)
                region_it.add_children(node)

        current_arg_name, current_node = region_iters[0].next()

    # Ensure regions are sorted in topological order
    for region in regions:
        region.reverse()


def apply_graph_deduplication(output_graph) -> None:  # type: ignore[no-untyped-def]
    duplicated_region_groups = output_graph.region_tracker.get_identical_regions(
        output_graph.graph
    )

    for region_group in duplicated_region_groups:
        region = region_group[0]
        (
            subgraph,
            node_ind_arg_inds,
            inds_with_external_users,
        ) = create_subgraph(region)
        sub_gm = torch.fx.GraphModule(output_graph.nn_modules, subgraph)
        subgraph_name = output_graph.install_subgraph("subgraph", sub_gm)
        with output_graph.graph.inserting_before():
            get_subgraph_node = output_graph.graph.create_node(
                "get_attr", subgraph_name, (), {}
            )
        for region in region_group:
            replace_region_with_subgraph(
                output_graph.graph,
                region,
                get_subgraph_node,
                node_ind_arg_inds.keys(),
                inds_with_external_users,
                subgraph,
            )


def replace_region_with_subgraph(
    graph: torch.fx.Graph,
    region: Region,
    get_subgraph_node: Node,
    node_ind_arg_ind: Iterable[Tuple[int, int]],
    inds_with_external_users: List[int],
    subgraph: torch.fx.Graph,
) -> None:
    sub_args = []
    for node_ind, arg_ind in node_ind_arg_ind:
        node = region[node_ind]
        flattened_args_kwargs, _ = tree_flatten((node.args, node.kwargs))
        sub_args.append(flattened_args_kwargs[arg_ind])

    invoke_args = (get_subgraph_node, "hi", tuple(sub_args))

    earliest_region_node = region[0]
    with graph.inserting_before(earliest_region_node):
        invoke_subgraph_node = graph.create_node(
            "call_function", torch.ops.higher_order.invoke_subgraph, invoke_args, {}
        )
        for ind, external_user_ind in enumerate(inds_with_external_users):
            node = region[external_user_ind]
            subgraph_output = graph.create_node(
                "call_function", operator.getitem, (invoke_subgraph_node, ind), {}
            )
            node.replace_all_uses_with(subgraph_output)

        # Erase in reverse topological order
        for node in reversed(region):
            graph.erase_node(node)


def get_external_inputs(
    region: Region,
) -> Tuple[DefaultDict[Node, List[Tuple[int, int]]], Set[Node]]:
    external_node_to_indices = defaultdict(list)
    nodes_unique = set(region)
    external_nodes = set()
    for node_ind, node in enumerate(region):
        flattened_args_kwargs, _ = tree_flatten((node.args, node.kwargs))
        for arg_ind, in_node in enumerate(flattened_args_kwargs):
            if in_node not in nodes_unique and isinstance(in_node, Node):
                external_node_to_indices[in_node].append((node_ind, arg_ind))
                external_nodes.add(in_node)

    return external_node_to_indices, external_nodes  # type: ignore[return-value]


def get_inds_with_external_users(region: Region) -> List[int]:
    inds_to_output = []
    for ind, node in enumerate(region):
        for user in node.users:
            if user not in region:
                inds_to_output.append(ind)
                break

    return inds_to_output


def copy_nodes_and_remap_inputs(
    subgraph: torch.fx.Graph, region: Region
) -> Dict[Tuple[int, int], Any]:
    external_inputs_to_indices, external_inputs = get_external_inputs(region)
    indices_to_placeholder_ind: Dict[Tuple[int, int], Any] = {}
    region_to_subgraph_node = {}
    for arg_ind, node in enumerate(external_inputs):
        placeholder = subgraph.placeholder(f"subgraph_input_{node.name}")
        region_to_subgraph_node[node] = placeholder
        arg_indices = external_inputs_to_indices[node]
        for index_pair in arg_indices:
            indices_to_placeholder_ind[index_pair] = None

    def map_arg(node: Node) -> Node:
        if node in region_to_subgraph_node:
            return region_to_subgraph_node[node]
        else:
            return node

    for node in region:
        subgraph_node = subgraph.node_copy(node, lambda old: map_arg(old))
        region_to_subgraph_node[node] = subgraph_node

    return indices_to_placeholder_ind


def create_subgraph_outputs(
    subgraph: torch.fx.Graph, inds_to_output: List[int]
) -> None:
    node_list = [n for n in subgraph.nodes if n.op not in ("placeholder", "output")]
    out_tup = tuple(node_list[ind] for ind in inds_to_output)
    subgraph.output(out_tup)


def create_subgraph(
    region: Region,
) -> Tuple[torch.fx.Graph, Dict[Tuple[int, int], Any], List[int]]:
    subgraph: torch.fx.Graph = torch.fx.Graph()
    node_ind_input_inds = copy_nodes_and_remap_inputs(subgraph, region)
    inds_with_external_users = get_inds_with_external_users(region)
    create_subgraph_outputs(subgraph, inds_with_external_users)
    return subgraph, node_ind_input_inds, inds_with_external_users
