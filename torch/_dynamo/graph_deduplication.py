import operator
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Set, Tuple

import torch.fx
from torch.utils._pytree import tree_flatten

from .graph_region_tracker import Node, Region


def apply_graph_deduplication(output_graph) -> None:  # type: ignore[no-untyped-def]
    duplicated_region_groups = output_graph.region_tracker.get_identical_regions(
        output_graph.graph
    )

    for region_group in duplicated_region_groups:
        inds_with_external_users = _get_all_output_indices(region_group)
        region = region_group[0]
        (
            subgraph,
            node_ind_arg_inds,
        ) = _create_subgraph(region, inds_with_external_users)
        sub_gm = torch.fx.GraphModule(output_graph.nn_modules, subgraph)
        subgraph_name = output_graph.install_subgraph("subgraph", sub_gm)
        with output_graph.graph.inserting_before():
            get_subgraph_node = output_graph.graph.create_node(
                "get_attr", subgraph_name, (), {}
            )
        for region in region_group:
            _replace_region_with_subgraph(
                output_graph.graph,
                region,
                get_subgraph_node,
                node_ind_arg_inds.keys(),
                inds_with_external_users,
                subgraph,
                subgraph_name,
            )


def _replace_region_with_subgraph(
    graph: torch.fx.Graph,
    region: Region,
    get_subgraph_node: Node,
    node_ind_arg_ind: Iterable[Tuple[int, int]],
    inds_with_external_users: List[int],
    subgraph: torch.fx.Graph,
    subgraph_name: str,
) -> None:
    sub_args = []
    for node_ind, arg_ind in node_ind_arg_ind:
        node = region[node_ind]
        flattened_args_kwargs, _ = tree_flatten((node.args, node.kwargs))
        sub_args.append(flattened_args_kwargs[arg_ind])

    invoke_args = (get_subgraph_node, subgraph_name, tuple(sub_args))

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


def _get_external_inputs(
    region: Region,
) -> DefaultDict[Node, List[Tuple[int, int]]]:
    external_node_to_indices = defaultdict(list)
    nodes_unique = set(region)
    for node_ind, node in enumerate(region):
        flattened_args_kwargs, _ = tree_flatten((node.args, node.kwargs))
        for arg_ind, in_node in enumerate(flattened_args_kwargs):
            if in_node not in nodes_unique and isinstance(in_node, Node):
                external_node_to_indices[in_node].append((node_ind, arg_ind))

    return external_node_to_indices


def _get_all_output_indices(regions: List[Region]) -> List[int]:
    # Scan all regions to get the set of all possible output nodes indices in the region
    # perhaps it's possible to get this info some other way?
    inds_with_external_users: Set[int] = set()
    for region in regions:
        _get_inds_with_external_users(region, inds_with_external_users)

    return sorted(inds_with_external_users)


def _get_inds_with_external_users(region: Region, inds_unique: Set[int]) -> None:
    for ind, node in enumerate(region):
        for user in node.users:
            if user not in region:
                if ind not in inds_unique:
                    inds_unique.add(ind)


def _copy_nodes_and_remap_inputs(
    subgraph: torch.fx.Graph, region: Region
) -> Dict[Tuple[int, int], Any]:
    external_inputs_to_indices = _get_external_inputs(region)
    indices_to_placeholder_ind: Dict[Tuple[int, int], Any] = {}
    region_to_subgraph_node = {}
    for arg_ind, node in enumerate(external_inputs_to_indices.keys()):
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


def _create_subgraph_outputs(
    subgraph: torch.fx.Graph, inds_to_output: List[int]
) -> None:
    node_list = [n for n in subgraph.nodes if n.op not in ("placeholder", "output")]
    out_tup = tuple(node_list[ind] for ind in inds_to_output)
    subgraph.output(out_tup)


def _create_subgraph(
    region: Region,
    inds_with_external_users: List[int],
) -> Tuple[torch.fx.Graph, Dict[Tuple[int, int], Any]]:
    subgraph: torch.fx.Graph = torch.fx.Graph()
    node_ind_input_inds = _copy_nodes_and_remap_inputs(subgraph, region)
    _create_subgraph_outputs(subgraph, inds_with_external_users)
    return subgraph, node_ind_input_inds
