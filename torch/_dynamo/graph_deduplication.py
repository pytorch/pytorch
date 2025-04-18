"""
This module implements graph deduplication functionality for TorchDynamo's optimization pipeline.
Graph deduplication identifies identical subgraphs in the computational graph and merges them
to reduce redundancy and improve performance. The process involves analyzing regions of the graph,
identifying structurally equivalent regions, and replacing them with a single shared implementation.
This optimization is particularly effective for models with repeated patterns or similar computational
structures across different parts of the network.
"""

import logging
import operator
from collections import defaultdict
from collections.abc import Generator, Iterable
from typing import Any, Optional

import torch
import torch.fx
from torch._dynamo import config
from torch._higher_order_ops.utils import has_potential_input_alias_or_mutation
from torch.utils._ordered_set import OrderedSet

from .graph_region_tracker import Node, Region
from .graph_utils import _detect_cycles, _flatten_args_kwargs


log = logging.getLogger(__name__)


def apply_graph_deduplication(output_graph) -> dict[str, torch.fx.GraphModule]:  # type: ignore[no-untyped-def]
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
    node_to_additional_deps = _populate_additional_deps(output_graph.graph)

    sub_gms: dict[str, torch.fx.GraphModule] = {}

    for region_group in duplicated_region_groups:
        inds_with_external_users = _get_all_output_indices(region_group)
        region = region_group[0]
        (
            subgraph,
            node_ind_arg_inds,
        ) = _create_subgraph(region, inds_with_external_users)

        # Ignore regions with no args for now, could they possibly be evaluated at compile time?
        if not list(node_ind_arg_inds):
            continue

        sub_gm = torch.fx.GraphModule(output_graph.nn_modules, subgraph)
        subgraph_name = output_graph.install_subgraph("subgraph", sub_gm)
        sub_gms[subgraph_name] = sub_gm
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
                node_to_additional_deps,
            )

    _stable_topological_sort(output_graph.graph, node_to_additional_deps)
    return sub_gms


def _replace_region_with_subgraph(
    graph: torch.fx.Graph,
    region: Region,
    get_subgraph_node: Node,
    node_ind_arg_ind: Iterable[tuple[int, int]],
    inds_with_external_users: list[int],
    sub_gm: torch.fx.GraphModule,
    subgraph_name: str,
    node_to_additional_deps: dict[torch.fx.Node, list[torch.fx.Node]],
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

    from torch._inductor.pattern_matcher import stable_topological_sort

    invoke_subgraph_node = graph.create_node(
        "call_function", torch.ops.higher_order.invoke_subgraph, invoke_args, {}
    )
    for ind, external_user_ind in enumerate(inds_with_external_users):
        node = region[external_user_ind]
        subgraph_output = graph.create_node(
            "call_function", operator.getitem, (invoke_subgraph_node, ind), {}
        )
        node.replace_all_uses_with(subgraph_output, propagate_meta=True)

    # Erase in reverse topological order
    for node in reversed(region):
        graph.erase_node(node)
        node_to_additional_deps.pop(node)
        for dep_list in node_to_additional_deps.values():
            try:
                dep_list.remove(node)
            except ValueError:
                pass

    if config.graph_deduplication_lint:
        _detect_cycles(graph)
        stable_topological_sort(graph)
        graph.lint()

    if config.graph_deduplication_lint:
        graph.lint()


def _get_external_inputs(
    region: Region,
) -> dict[Node, tuple[int, int]]:
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


def _get_all_output_indices(regions: list[Region]) -> list[int]:
    # Scan all regions to get the set of all possible output nodes indices in the region
    # perhaps we can record this information during region creation for more efficiency?
    inds_with_external_users: set[int] = set()
    for region in regions:
        _get_inds_with_external_users(region, inds_with_external_users)

    return sorted(inds_with_external_users)


def _get_inds_with_external_users(region: Region, inds_unique: set[int]) -> None:
    for ind, node in enumerate(region):
        for user in node.users:
            if user not in region:
                if ind not in inds_unique:
                    inds_unique.add(ind)


def _copy_nodes_and_remap_inputs(
    subgraph: torch.fx.Graph, region: Region
) -> dict[tuple[int, int], Any]:
    external_inputs_to_indices = _get_external_inputs(region)
    indices_to_placeholder_ind: dict[tuple[int, int], Any] = {}
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
    subgraph: torch.fx.Graph, inds_to_output: list[int]
) -> None:
    node_list = [n for n in subgraph.nodes if n.op not in ("placeholder", "output")]
    out_tup = tuple(node_list[ind] for ind in inds_to_output)
    subgraph.output(out_tup)


def _create_subgraph(
    region: Region,
    inds_with_external_users: list[int],
) -> tuple[torch.fx.Graph, dict[tuple[int, int], Any]]:
    subgraph: torch.fx.Graph = torch.fx.Graph()
    node_ind_input_inds = _copy_nodes_and_remap_inputs(subgraph, region)
    _create_subgraph_outputs(subgraph, inds_with_external_users)
    return subgraph, node_ind_input_inds


def _args(
    n: torch.fx.Node,
    node_to_additional_deps: Optional[dict[torch.fx.Node, list[torch.fx.Node]]] = None,
) -> list[torch.fx.node.Argument]:
    if node_to_additional_deps is None:
        node_to_additional_deps = {}

    args: list[torch.fx.node.Argument] = []
    torch.fx.map_arg((n.args, n.kwargs), args.append)
    if n in node_to_additional_deps:
        args.extend(node_to_additional_deps[n])
    return args


def _stable_topological_sort(
    graph: torch.fx.Graph,
    node_to_additional_deps: dict[torch.fx.Node, list[torch.fx.Node]],
) -> None:
    # Nodes are in exactly one of these three collections:

    # - Nodes in `pending` are waiting to be processed (in reverse order):
    pending = list(reversed(graph.nodes))

    # - Nodes in `ready` have been processed and are already in the correct
    #   order.
    ready = OrderedSet[torch.fx.Node]()

    # - `waiting` is a mapping from a dependency to nodes which depend on that
    #   dependency.
    waiting = defaultdict(list)

    # The cursor indicates the last processed node so we can add new nodes
    # after it.
    cursor = None
    while pending:
        node = pending.pop()
        waiting_for = [
            x for x in _args(node, node_to_additional_deps) if x not in ready
        ]
        if waiting_for:
            # We have unprocessed input nodes. Might as well wait for the last
            # arg so an already sorted list will only recheck this node once.
            waiting[waiting_for[-1]].append(node)
        else:
            ready.add(node)
            if cursor and cursor.next is not node:
                cursor.append(node)
            cursor = node
            # Mark the nodes that have been waiting for this node to finish as
            # ready to check again.
            pending.extend(reversed(waiting.pop(node, ())))

    assert not waiting and len(ready) == len(graph.nodes)


def _populate_additional_deps(
    graph: torch.fx.Graph,
) -> dict[torch.fx.Node, list[torch.fx.Node]]:
    import torch.amp

    node_to_additional_deps: dict[torch.fx.Node, list[torch.fx.Node]] = defaultdict(
        list
    )
    all_nodes = list(graph.nodes)

    # These are targets of the nodes which need to stay in the same relative place in the graph
    global_state_targets = {torch.amp._enter_autocast, torch.amp._exit_autocast}
    all_nodes_dep_on: list[torch.fx.Node] = []

    def prev_cur_nodes(
        all_nodes: list[torch.fx.Node],
    ) -> Generator[tuple[list[torch.fx.Node], torch.fx.Node]]:
        prev_nodes: list[torch.fx.Node] = []
        next_nodes = list(reversed(all_nodes))

        while next_nodes:
            cur_node = next_nodes.pop()
            yield prev_nodes, cur_node
            prev_nodes.append(cur_node)

    for prev_nodes, cur_node in prev_cur_nodes(all_nodes):
        args_unique = _args(cur_node)
        additional_deps = node_to_additional_deps[cur_node]
        additional_deps.extend(n for n in all_nodes_dep_on if n not in args_unique)
        if cur_node.target in global_state_targets:
            additional_deps.extend(n for n in prev_nodes if n not in args_unique)
            all_nodes_dep_on.append(cur_node)

    return node_to_additional_deps
