import copy
from queue import SimpleQueue
from typing import List, Dict

import torch.fx
from torch.fx.graph_module import GraphModule
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx.passes.tools_common import NodeList, NodeSet, legalize_graph
from torch.fx.passes.split_utils import HolderModule

def topo_sort(nodes: NodeList) -> NodeList:
    # sort nodes according to the topological order
    indegree_map = {node : 0 for node in nodes}
    candidates: SimpleQueue = SimpleQueue()

    for node in nodes:
        for n in node.all_input_nodes:
            if n in indegree_map:
                indegree_map[node] += 1
        if indegree_map[node] == 0:
            candidates.put(node)

    sorted_nodes: NodeList = list()
    while not candidates.empty():
        node = candidates.get()
        sorted_nodes.append(node)

        for n in node.users:
            if n in indegree_map:
                indegree_map[n] -= 1
                if indegree_map[n] == 0:
                    candidates.put(n)

    assert len(nodes) == len(sorted_nodes), "topological sorted nodes doesn't have same length as input nodes"

    return sorted_nodes


def validate_partition(partition: NodeList) -> bool:
    # verify the partition does't form a dependency cycle in the original graph
    # returns True for valid partition, False for invalid

    partition_set = set(partition)

    outputs: NodeList = list()
    for node in partition_set:
        for user_node in node.users:
            if user_node not in partition_set:
                # external user node, need to expose as an output
                outputs.append(user_node)

    # perform DFS on the parition outputs
    # if it reaches a node within the partition, then it found a cycle
    visited: NodeSet = set()

    def dfs_find_cycle(node):
        if node in partition_set:
            return True  # found cycle, return

        visited.add(node)
        for user_node in node.users:
            if user_node not in visited:
                if dfs_find_cycle(user_node):
                    return True
        return False

    for output_node in outputs:
        if dfs_find_cycle(output_node):
            return False

    return True


# TODO: copied from split_by_tags's impl, refactor split_by_tags to use this function
def copy_module_attributes(gm: GraphModule, subgraph: Graph) -> HolderModule:
    # Loop through all module calls (call_module) and param fetches (get_attr)
    # in this component, creating HolderModules as necessary to match the path.
    # e.g. if in the original module there's a get_attr node fetches "conv.weight".
    # We create a HolderModule as root -> add a HolderModule named "conv" ->
    # make "weight" a attribute of "conv" HolderModule and point to conv.weight in
    # the original module.
    submodule = HolderModule({})
    for n in subgraph.nodes:
        if n.op not in ("call_module", "get_attr"):
            continue

        target = n.target
        assert isinstance(target, str)
        target_name_parts = target.split(".")
        curr = submodule
        orig_gm = gm

        for name in target_name_parts[:-1]:
            if not hasattr(curr, name):
                curr.add_module(name, HolderModule({}))

            curr = getattr(curr, name)
            orig_gm = getattr(orig_gm, name)

        leaf_node_name = target_name_parts[-1]
        leaf_node = getattr(orig_gm, leaf_node_name)

        # Relies on custom __setattr__ magic.
        setattr(curr, leaf_node_name, leaf_node)

    return submodule

def fuse_partition(gm: GraphModule,
                   nodes: NodeList,
                   partition_name: str) -> GraphModule:
    # returns a graph module that is a copy of `nodes` in gm
    # assumption: nodes are already sorted in topo order

    for node in nodes:
        assert node.graph.owning_module is gm, f"{node} doesn't belong to passed in graph module {gm._get_name()}"
        assert not node._erased, f"{node} has been removed from owning graph"
        assert node in gm.graph.nodes, f"{node} is not found in graph module {gm._get_name()}"

    # validates partition doesn't introduce dependency circles in the graph
    assert validate_partition(nodes), "Invalid partition, found dependency cycles"

    subgraph = Graph()

    node_to_placeholder: Dict[Node, Node] = {}  # mapping of nodes from old graph to placeholder in new graph
    node_map: Dict[Node, Node] = {}       # mapping of nodes from old graph to new graph

    # handles inputs throught graph.node_copy's arg_transform functions
    def remap_inputs(x):
        if x.op == "get_attr":
            # TODO: do we really need copy the get_attr node into the graph?
            # do something here
            pass

        if x in nodes:
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
            if user_node not in nodes:
                # external user node, need to expose as an output
                output_mapping[node] = node_map[node]

    # outs contain nodes in the new subgraph
    original_outputs = tuple(output_mapping.keys())
    outs = tuple(output_mapping.values())

    # Take care of the args of FX output node. If there's a single
    # output then the output node args is like (output_single), else
    # if there're multiple outputs then the output node args is like
    # ((output_0, output_1, ...)).
    subgraph.output(outs[0] if len(outs) == 1 else outs)

    # lint to ensure correctness
    subgraph.lint()

    submodule = copy_module_attributes(gm, subgraph)

    sub_gm = GraphModule(submodule, subgraph, class_name=partition_name)

    # TODO: fix this
    sub_gm.name = partition_name
    sub_gm.orig_inputs = tuple(node_to_placeholder.keys())
    sub_gm.orig_outputs = original_outputs

    return sub_gm

def insert_subgm(gm, sub_gm, original_nodes):
    # assign sub_gm into gm
    setattr(gm, sub_gm.name, sub_gm)

    # Create a call_module node in main graph.
    module_node = gm.graph.call_module(
        sub_gm.name,
        args=sub_gm.orig_inputs,
        kwargs=None,
    )

    orig_outputs = sub_gm.orig_outputs

    if len(orig_outputs) == 1:
        # main_remapping[comp.orig_outputs[0]] = module_node
        orig_outputs[0].replace_all_uses_with(module_node)
    else:
        for i, out in enumerate(orig_outputs):
            # Use Proxy to record getitem access.
            proxy_out = torch.fx.Proxy(module_node)[i].node  # type: ignore[index]

            out.replace_all_uses_with(proxy_out)

    # erase original nodes in inversed topological order
    for node in reversed(original_nodes):
        gm.graph.erase_node(node)

    return gm

def fuse_by_partitions(gm: GraphModule, partitions: List[NodeList]) -> GraphModule:
    for partition_id, nodes in enumerate(partitions):
        sorted_nodes = topo_sort(nodes)

        partition_name = "fused_" + str(partition_id)
        sub_gm = fuse_partition(gm, sorted_nodes, partition_name)

        insert_subgm(gm, sub_gm, sorted_nodes)

    # topological sort original gm with newly created sub_gm
    legalize_graph(gm)

    return gm
