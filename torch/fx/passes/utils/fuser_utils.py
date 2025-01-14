# mypy: allow-untyped-defs
import copy
from queue import SimpleQueue
from typing import Dict, List, Optional as _Optional, Tuple

import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.fx.passes.tools_common import legalize_graph, NodeList, NodeSet
from torch.fx.passes.utils import lift_subgraph_as_module


@compatibility(is_backward_compatible=False)
def topo_sort(nodes: NodeList) -> NodeList:
    # sort nodes according to the topological order
    indegree_map = dict.fromkeys(nodes, 0)
    candidates: SimpleQueue = SimpleQueue()

    for node in nodes:
        for n in node.all_input_nodes:
            if n in indegree_map:
                indegree_map[node] += 1
        if indegree_map[node] == 0:
            candidates.put(node)

    sorted_nodes: NodeList = []
    while not candidates.empty():
        node = candidates.get()
        sorted_nodes.append(node)

        for n in node.users:
            if n in indegree_map:
                indegree_map[n] -= 1
                if indegree_map[n] == 0:
                    candidates.put(n)

    assert len(nodes) == len(
        sorted_nodes
    ), "topological sorted nodes doesn't have same length as input nodes"

    return sorted_nodes


@compatibility(is_backward_compatible=False)
def validate_partition(partition: NodeList) -> bool:
    # verify the partition does't form a dependency cycle in the original graph
    # returns True for valid partition, False for invalid

    partition_set = set(partition)

    outputs: NodeList = []
    for node in partition_set:
        for user_node in node.users:
            if user_node not in partition_set:
                # external user node, need to expose as an output
                outputs.append(user_node)

    # Perform BFS on the partition outputs.
    # If it reaches a node within the partition, then it found a cycle.
    # This function takes the ownership of `root_nodes` and may modify it.
    def bfs_find_cycle(root_nodes: NodeList) -> bool:
        # Set used to exclude nodes that have already been visited.
        # If a node has been visited, that node and all its children have
        # been checked for cycles.
        visited: NodeSet = set()

        # Start with `root_nodes` and traverse through (toward child nodes)
        # their connected sub-graph. Nodes in `visited` won't be added
        # to `queue` again.
        queue: NodeList = root_nodes
        while queue:
            current = queue.pop()
            visited.add(current)
            if current in partition_set:
                # Started from partition's `output` nodes, and reached
                # another node in partition. Cycle!
                return True
            for user_node in current.users:
                if user_node in visited:
                    continue
                queue.append(user_node)
        # `root_nodes` don't cause cycle.
        return False

    # Use all output nodes as roots to traverse
    # the graph to check cycles.
    if bfs_find_cycle(outputs):
        return False

    return True


@compatibility(is_backward_compatible=False)
def fuse_as_graphmodule(
    gm: GraphModule,
    nodes: NodeList,
    module_name: str,
    partition_lookup_table: _Optional[Dict[Node, None]] = None,
    *,
    always_return_tuple: bool = False,
) -> Tuple[GraphModule, Tuple[Node, ...], Tuple[Node, ...]]:
    """
    Fuse nodes in graph_module into a GraphModule.

    Args:
        gm (GraphModule): target graph_module

        nodes (List[Node]): list of nodes in `gm` to fuse, where the node must be topologically sorted

        module_name: class name for the fused GraphModule

        partition_lookup_table (Optional[Dict[Node, None]]): optional dict of nodes to speed up lookup

        always_return_tuple (bool): whether to always return a tuple, even if there is only one output

    Returns:
        fused_gm (GraphModule): fused graph module, where its node is a copy of `nodes` in `gm`

        original_inputs (Tuple[Node, ...]): input nodes to `nodes` in original `gm`

        original_outputs (Tuple[Node, ...]): consumer nodes of `nodes` in original `gm`

    """

    # assumption: nodes are already sorted in topo order

    for node in nodes:
        assert (
            node.graph.owning_module is gm
        ), f"{node} doesn't belong to passed in graph module {gm._get_name()}"
        assert not node._erased, f"{node} has been removed from owning graph"
        assert (
            node in gm.graph._find_nodes_lookup_table
        ), f"{node} is not found in graph module {gm._get_name()}"

    # validates partition doesn't introduce dependency circles in the graph
    assert validate_partition(nodes), "Invalid partition, found dependency cycles"

    # if no dict of partition nodes is provided, reconstruct it by nodes list to reduce lookup time
    if partition_lookup_table is None:
        partition_lookup_table = dict.fromkeys(nodes)

    subgraph = Graph()

    node_to_placeholder: Dict[
        Node, Node
    ] = {}  # mapping of nodes from old graph to placeholder in new graph
    node_map: Dict[Node, Node] = {}  # mapping of nodes from old graph to new graph

    # handles inputs through graph.node_copy's arg_transform functions
    def remap_inputs(x):
        if x.op == "get_attr":
            # TODO: do we really need copy the get_attr node into the graph?
            # do something here
            pass

        if x in partition_lookup_table:
            # x is inside subgraph, return the copied node
            # the node should have been copied aleady, as we are copying graph in the topological order
            return node_map[x]

        if x not in node_to_placeholder:
            # x is not in subgraph, create a new placeholder for subgraph
            placeholder_node = subgraph.placeholder(x.name, type_expr=x.type)
            # copy all meta fields, even if some fields might be irrelvant for the placeholder node
            placeholder_node.meta = copy.copy(x.meta)
            node_to_placeholder[x] = placeholder_node

        return node_to_placeholder[x]

    # copy nodes in topological order
    for node in nodes:
        new_node = subgraph.node_copy(node, remap_inputs)
        node_map[node] = new_node

    # handles outputs
    output_mapping: Dict[Node, Node] = {}  # mapping from old output to new outputs

    for node in nodes:
        for user_node in node.users:
            if user_node not in partition_lookup_table:
                # external user node, need to expose as an output
                output_mapping[node] = node_map[node]

    # outs contain nodes in the new subgraph
    outs = tuple(output_mapping.values())

    if always_return_tuple:
        # always return a tuple, even if there is only one output
        subgraph.output(outs)
    else:
        # If there's a single output then return it directly, otherwise return a tuple.
        subgraph.output(outs[0] if len(outs) == 1 else outs)

    # lint to ensure correctness
    subgraph.lint()
    fused_gm: GraphModule
    fused_gm, _ = lift_subgraph_as_module(
        gm, subgraph, comp_name="", class_name=module_name
    )

    # sub_gm's input nodes in the original module
    original_inputs: Tuple[Node, ...] = tuple(node_to_placeholder.keys())

    # sub_gm's outputs node in the original module
    original_outputs: Tuple[Node, ...] = tuple(output_mapping.keys())

    return fused_gm, original_inputs, original_outputs


@compatibility(is_backward_compatible=False)
def insert_subgm(
    gm: GraphModule,
    sub_gm: GraphModule,
    orig_inputs: Tuple[Node, ...],
    orig_outputs: Tuple[Node, ...],
):
    # add sub_gm into gm
    submodule_name = sub_gm.__class__.__name__
    gm.add_submodule(submodule_name, sub_gm)

    # Create a call_module node in main graph.
    module_node = gm.graph.call_module(submodule_name, args=orig_inputs, kwargs=None)

    output_node = sub_gm.graph.output_node()
    if len(orig_outputs) == 1 and not isinstance(output_node.args[0], tuple):
        # main_remapping[comp.orig_outputs[0]] = module_node
        orig_outputs[0].replace_all_uses_with(module_node, propagate_meta=True)
    else:
        for i, orig_output in enumerate(orig_outputs):
            # Use Proxy to record getitem access.
            proxy_out = torch.fx.Proxy(module_node)[i].node  # type: ignore[index]
            orig_output.replace_all_uses_with(proxy_out, propagate_meta=True)

        module_node.meta["val"] = tuple(
            orig_output.meta.get("val", None) for orig_output in orig_outputs
        )
    return gm


@compatibility(is_backward_compatible=False)
def erase_nodes(gm: GraphModule, nodes: NodeList):
    # erase original nodes in inversed topological order
    for node in reversed(nodes):
        gm.graph.erase_node(node)


@compatibility(is_backward_compatible=False)
def fuse_by_partitions(
    gm: GraphModule,
    partitions: List[Dict[Node, None]],
    prefix: str = "fused_",
    always_return_tuple: bool = False,
) -> GraphModule:
    for partition_id, partition in enumerate(partitions):
        sorted_nodes = topo_sort(list(partition))

        submodule_name = prefix + str(partition_id)
        sub_gm, orig_inputs, orig_outputs = fuse_as_graphmodule(
            gm,
            sorted_nodes,
            submodule_name,
            partition,
            always_return_tuple=always_return_tuple,
        )

        insert_subgm(gm, sub_gm, orig_inputs, orig_outputs)

        erase_nodes(gm, sorted_nodes)

    # topological sort original gm with newly created sub_gm
    legalize_graph(gm)

    return gm
