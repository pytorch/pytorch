import logging
import operator
from typing import Any, Dict, Iterable, List, Set, Tuple

import torch.fx
from torch._higher_order_ops.utils import has_potential_input_alias_or_mutation
from torch.utils._pytree import tree_flatten

from .graph_region_tracker import Node, Region


log = logging.getLogger(__name__)


def apply_graph_deduplication(output_graph) -> Dict[Node, Node]:  # type: ignore[no-untyped-def]
    """
    This is the main entry point for applying the graph deduplication pass. \
Deduplication occurs in two phases:
    1. Subgraph creation:
        Subgraph creation works by taking one representative region from each region \
group and creating a subgraph from it, which will then be used to replace all regions \
in the group. This is implemented by first copying all nodes of the region to the new \
subgraph and then finding all inputs which are not within the region and creating placeholders \
for them. For the outputs, all regions in a region group need to be scanned to ensure the \
largest set of outputs is found, and then an output node is created which returns \
a tuple of all outputs.

    2. Graph replacement:
        To replace each region with the extracted subgraph, the node index in the region \
and argument index within the node's flattened args and kwargs are recorded once during \
subgraph creation. This allows us to determine which (external to the region) nodes and \
in which order these nodes are passed as inputs. For the outputs, getitem nodes are created \
for each output, and all nodes in the region with external outputs are replaced by the proper \
getitem node. Finally, all original nodes are erased (there should be no uses of these \
left in the graph).

The deduplication mutates the output_graph argument in place.

Returns a mapping of nodes to their subgraph output replacement node to remap outputs
when they are created in output_graph.
    """
    duplicated_region_groups = output_graph.region_tracker.get_identical_regions(
        output_graph.graph
    )

    # Used to track which nodes were replaced with subgraph outputs
    # today, we have to register the new subgraph submodules before the
    # graph outputs have been created, so we pass the replacement mapping
    # back to output graph to do the replacements at the site of output creation
    output_replacements: Dict[Node, Node] = {}
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
                sub_gm,
                subgraph_name,
                output_replacements,
            )

    return output_replacements


# flattens with support for slices
# Note: a better way to do this would
# be register/unregister slices as pytree nodes
# but there is no unregister API in the pytorch
# pytree impl
def _flatten_args_kwargs(args: Any) -> List[Node]:
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


def _replace_region_with_subgraph(
    graph: torch.fx.Graph,
    region: Region,
    get_subgraph_node: Node,
    node_ind_arg_ind: Iterable[Tuple[int, int]],
    inds_with_external_users: List[int],
    sub_gm: torch.fx.GraphModule,
    subgraph_name: str,
    output_replacements: Dict[Node, Node],
) -> None:
    sub_args = []
    for node_ind, arg_ind in node_ind_arg_ind:
        node = region[node_ind]
        flattened_args_kwargs = _flatten_args_kwargs((node.args, node.kwargs))
        sub_args.append(flattened_args_kwargs[arg_ind])

    invoke_args = (get_subgraph_node, subgraph_name, tuple(sub_args))
    fake_inputs = [node.meta["example_value"] for node in sub_args]

    if has_potential_input_alias_or_mutation(sub_gm, fake_inputs):
        log.debug(
            "NYI: Failed to substitute region %s due to input alias or mutation",
            region,
        )
        return

    latest_region_node = region[-1]
    with graph.inserting_after(latest_region_node):
        invoke_subgraph_node = graph.create_node(
            "call_function", torch.ops.higher_order.invoke_subgraph, invoke_args, {}
        )
        with graph.inserting_after(invoke_subgraph_node):
            for ind, external_user_ind in enumerate(inds_with_external_users):
                node = region[external_user_ind]
                subgraph_output = graph.create_node(
                    "call_function", operator.getitem, (invoke_subgraph_node, ind), {}
                )
                output_replacements[node] = subgraph_output
                node.replace_all_uses_with(subgraph_output, propagate_meta=True)

        # Erase in reverse topological order
        for node in reversed(region):
            graph.erase_node(node)


def _get_external_inputs(
    region: Region,
) -> Dict[Node, Tuple[int, int]]:
    external_node_to_indices = dict()
    region_unique = set(region)
    for node_ind, node in enumerate(region):
        flattened_args_kwargs = _flatten_args_kwargs((node.args, node.kwargs))
        for arg_ind, in_node in enumerate(flattened_args_kwargs):
            if (
                isinstance(in_node, Node)
                and in_node not in region_unique
                and in_node not in external_node_to_indices
            ):
                external_node_to_indices[in_node] = (node_ind, arg_ind)

    return external_node_to_indices


def _get_all_output_indices(regions: List[Region]) -> List[int]:
    # Scan all regions to get the set of all possible output nodes indices in the region
    # perhaps we can record this information during region creation for more efficiency?
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
    for node in external_inputs_to_indices.keys():
        placeholder = subgraph.placeholder(f"subgraph_input_{node.name}")
        region_to_subgraph_node[node] = placeholder
        arg_indices = external_inputs_to_indices[node]
        # Note: insertion order matches the order in which placeholders were created
        # for the calling convention of the subgraph
        indices_to_placeholder_ind[arg_indices] = None

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
