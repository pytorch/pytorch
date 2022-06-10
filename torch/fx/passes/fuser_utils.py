from queue import SimpleQueue
from typing import List, Optional, Dict

import torch.fx
from torch.fx.graph import map_arg
from torch.fx.passes.tools_common import NodeList, NodeSet, legalize_graph
from torch.fx.passes.split_utils import HolderModule

def topo_sort(nodes: NodeList) -> NodeList:
    # sort nodes according to the topological order
    indegree_map = {node : 0 for node in nodes}
    candidates = SimpleQueue()

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

    return sorted_nodes

# TODO: copied from split_by_tags's impl, refactor split_by_tags to use this function
def copy_module_attributes(gm: torch.fx.GraphModule, subgraph: torch.fx.Graph) -> HolderModule:
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

def fuse_partition(gm: torch.fx.GraphModule,
                   nodes: NodeList,
                   partition_name: str) -> torch.fx.GraphModule:
    # returns a graph module that is a copy of `nodes` in gm
    # assumption: nodes are already sorted in topo order

    for node in nodes:
        assert node.graph.owning_module is gm, f"{node} doesn't belong to passed in graph module {gm._get_name()}"
        assert not node._erased, f"{node} has been removed from owning graph"
        assert node in gm.graph.nodes, f"{node} is not found in graph module {gm._get_name()}"

    # TODO: validate partition
    # - partition doesn't introduce circles in the graph

    subgraph = torch.fx.Graph()

    node_to_placeholder = {}  # mapping of nodes from old graph to placeholder in new graph
    node_map = {}       # mapping of nodes from old graph to new graph

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
            node_to_placeholder[x] = subgraph.placeholder(x.name, type_expr=x.type)

        return node_to_placeholder[x]

    # copy nodes in topological order
    for node in nodes:
        new_node = subgraph.node_copy(node, remap_inputs)
        node_map[node] = new_node

    # handles outputs
    output_mapping = {}  # mapping from old output to new outputs

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

    sub_gm = torch.fx.GraphModule(submodule, subgraph, class_name=partition_name)

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

def fuse_by_partitions(gm: torch.fx.GraphModule, partitions: List[NodeList]) -> torch.fx.GraphModule:
    for partition_id, nodes in enumerate(partitions):
        partition_name = "fused_" + str(partition_id)

        sorted_nodes = topo_sort(nodes)

        sub_gm = fuse_partition(gm, sorted_nodes, partition_name)

        # print(partition_name)
        # print(sub_gm.graph)

        insert_subgm(gm, sub_gm, sorted_nodes)

    # print("before")
    # print(gm)

    legalize_graph(gm)

    # print("after")
    # print(gm)

    return gm
