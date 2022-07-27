import operator
import math
import copy
import os
from typing import List, Tuple, Dict, Set

import torch
import torch.fx as fx
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.backends.nvfuser import NvFuserOperatorSupport
from torch.fx.passes.tools_common import legalize_graph

from .utilities import _size_of, ban_recomputation

REMATERIALIZATION_DEBUG = bool(os.environ.get("REMATERIALIZATION_DEBUG", False))

num_group_remat = 0  # used for analytical purpose
memory_reduced = 0
num_node_pairs = 0


def is_fused_node(node) -> bool:
    return node.op == "call_module" and "fused_" in node.target


def has_remat_node(module_node: fx.Node, fused_graph: fx.GraphModule) -> bool:
    """
    Return True if the submodule of ``fused_graph`` corresponding to ``module_node``
    has at least one node that can be rematerialized.
    """
    module = getattr(fused_graph, module_node.name)
    try_remat = False
    for node in module.graph.nodes:
        if node.target != operator.getitem and node.op == "call_function" and not ban_recomputation(node):
            try_remat = True
            break
    return try_remat


def try_remat(node: fx.Node, fused_graph: fx.GraphModule) -> bool:
    """
    Return True if there might be rematerialization opportunities in the submodule of ``fused_graph``
    corresponding to ``node``.
    """
    return is_fused_node(node) and has_remat_node(node, fused_graph)


def get_users(node: fx.Node) -> Set[fx.Node]:
    """
    Return the users of a node
    A user might directly take in the node as an arg or
    use the output of node through getitem
    """
    users = set()
    for user_node in node.users:
        if user_node.target == operator.getitem:
            users = users.union(set(user_node.users.keys()))
        elif user_node.op != 'output':
            users.add(user_node)
    return users


def get_fused_node_pairs(fused_graph: fx.GraphModule) -> List[Tuple[fx.Node, fx.Node]]:
    """
    Return a list of pairs of fused nodes that are (parent, child) relationship in fused_graph.
    the two (parent, child) nodes might have an getitem node between them
    """
    fused_node_pairs = []
    for node in fused_graph.graph.nodes:
        if(try_remat(node, fused_graph)):
            users = get_users(node)
            pairs = [(node, user_node) for user_node in users if (try_remat(user_node, fused_graph))]
            fused_node_pairs.extend(pairs)
    return fused_node_pairs


def get_weight(node: fx.Node) -> int:
    weight = 0
    if 'tensor_meta' in node.meta:
        weight = _size_of(node.meta['tensor_meta'])
    return weight


def get_name_to_args_map(node_orig: fx.Node, gm: fx.GraphModule) -> Dict[str, fx.Node]:
    """
    Return a map from placeholder name in gm to node_orig.args

    ``node_orig`` is the node corresponding to submodule ``gm`` in some fused_graph.
    """
    placeholder_map = {}
    loc = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            placeholder_map[node.name] = node_orig.args[loc]
            loc += 1
    return placeholder_map


def get_nx_node_name(node_name: str) -> str:
    if node_name.endswith("_in"):
        return node_name[:-3]
    elif node_name.endswith("_out"):
        return node_name[:-4]
    raise Exception("node name is not _in or _out, " + node_name)


def get_cut_nodes_from_partition(partition: Tuple[Set[str], Set[str]], nx_graph) -> Set[str]:
    """
    Return the cut nodes from the partition. Cut nodes are the nodes reachable from the root node
    and have outgoing edges to nodes in the non-reachable partition.

    Args:
    ``partition`` is a tuple of set of nodes in nx_graph.
    ``nx_graph`` is a networkx graph.
    """
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, nx_graph[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    cut_nodes = set()
    for node_in, _ in cutset:
        cut_nodes.add(get_nx_node_name(node_in))
    return cut_nodes


def order_topologically(nodes: List[fx.Node], gm: fx.GraphModule) -> List[fx.Node]:
    node_order_dict = {}
    rank = 0
    for n in gm.graph.nodes:
        node_order_dict[n.name] = rank
        rank += 1

    nodes = sorted(nodes, key=lambda x: node_order_dict[x])
    return nodes


def get_output_node_args(node: fx.Node) -> Tuple[fx.Node]:
    if type(node.args[0]) is not tuple:
        return node.args
    return node.args[0]


def get_user_name_to_user_map(output_node_in_module: fx.Node, fused_node: fx.Node) -> Dict[str, fx.Node]:
    """
    Return a dictionary from the names of the user nodes of ``fused_node``
    to ``output_node_in_module``'s args. The user nodes here are the nodes that
    uses the output of ``fused_node`` through getitem nodes.

    ``output_node_in_module`` is the output node of the submodule corresponding to ``fused_node`` in a fused graph.
    """
    assert(output_node_in_module.op == "output")
    user_name_to_user_map = {}
    output_args = get_output_node_args(output_node_in_module)
    for user in fused_node.users:
        # can only do this for getitem users. might have a single add node that have two users
        if user.target != operator.getitem:
            break
        loc = user.args[1]
        if isinstance(output_args[loc], torch.fx.node.Node):
            user_name = output_args[loc].name
            user_name_to_user_map[user_name] = user
    return user_name_to_user_map


def add_new_outputs(node_pair: Tuple[fx.Node, fx.Node],
                    fused_graph: fx.GraphModule,
                    name_to_node: Dict[str, fx.Node],
                    module_origin: fx.GraphModule,
                    module_origin_new_outputs: List[fx.Node]):
    """
    Modify the outputs of ``module_origin`` to contain new outputs required after rematerialization.
    Add new getitem nodes in ``fused_graph`` if necessary.

    ``module_origin_new_outputs`` is a list of outputs required after rematerialization.
    """
    origin_placeholders = set(node for node in module_origin.graph.nodes if node.op == "placeholder")
    for node in module_origin.graph.nodes:
        if node.op == "output":
            old_args = get_output_node_args(node)
            module_origin_new_outputs = list(module_origin_new_outputs.difference(set(old_args)).difference(origin_placeholders))

            # need to change the user to use getitem if origin only has 1 output but now has more
            if(len(old_args) == 1 and len(module_origin_new_outputs) > 0):
                with fused_graph.graph.inserting_after(node_pair[0]):
                    new_node = fused_graph.graph.call_function(operator.getitem, args=(node_pair[0], 0, ))
                node_pair[0].replace_all_uses_with(new_node)
                new_node.args = (node_pair[0], 0, )
                name_to_node[node_pair[0].name] = new_node  # add new arg to dest placeholder map

            if len(module_origin_new_outputs) > 0:
                # need to add new ouputs to module_origin and new inputs to module_dest
                with fused_graph.graph.inserting_after(node_pair[0]):
                    for i in range(len(module_origin_new_outputs)):
                        new_node = fused_graph.graph.call_function(
                            operator.getitem, args=(node_pair[0], i + len(old_args), ))
                new_args = list(old_args) + module_origin_new_outputs
                module_origin.graph.erase_node(node)
                module_origin.graph.output(new_args[0] if len(new_args) == 1 else tuple(new_args))
            break
    module_origin.recompile()
    fused_graph.recompile()


def remove_unused_output(fused_graph: fx.GraphModule, module_origin: fx.GraphModule):
    """
    Remove the unused output of module_origin to write less

    Use None as a placeholder for removed output args because getitem will index into the outputs.
    """
    fused_graph.graph.eliminate_dead_code()
    used_inds = set()

    # need to modify the node in fused_graph, not the node passed in pairs
    for node in fused_graph.graph.nodes:
        if(node.name == module_origin._get_name()):
            for node_user in node.users:
                if node_user.target == operator.getitem:
                    used_inds.add(node_user.args[1])
            break

    for node in module_origin.graph.nodes:
        if node.op == "output":
            if (len(used_inds) == 0 and type(node.args[0] is not tuple)):  # only has a single output
                break
            new_args = []
            for i in range(len(node.args[0])):
                if i in used_inds:
                    new_args.append(node.args[0][i])  # still useful
                else:
                    new_args.append(None)  # no need to write out
            node.args = (tuple(new_args), )
            break
    module_origin.recompile()
    fused_graph.recompile()


def get_node_to_copy(non_reachable: Set[str], cut_nodes: Set[str]) -> Set[str]:
    """
    All nodes corresponding to vertices in non_reachable partition are recomputed
    (except the nodes corresponding to cut vertices, whose are placeholders).
    """
    node_to_copy = set()
    for node_name in non_reachable:
        if node_name == "sink":
            continue
        node_name = get_nx_node_name(node_name)
        node_to_copy.add(node_name)
    node_to_copy = node_to_copy.difference(cut_nodes)  # cut nodes are handeled separately as placeholders
    if REMATERIALIZATION_DEBUG:
        print(node_to_copy)
    return node_to_copy


def copy_nodes(node_pair: Tuple[fx.Node, fx.Node],
               fused_graph: fx.GraphModule,
               name_to_node: Dict[str, fx.Node],
               partition: Tuple[Set[str], Set[str]],
               cut_nodes: Set[str]):
    """
    Copy nodes in the reachable partition to module of node_pair[1].

    Let the fusion groups corresponding to node_pair be (A,B).

    Give a partition (reachable, non_reachable) of the nodes in fusion group A,
    we want to re-compute all nodes in non_reachable in fusion group B.

    We want to modify fusion group A such that:
    * A outputs all nodes corresponding to cut vertices
    * A does not output any node that has no user anymore

    We want to modify fusion group B such that:
    * All nodes corresponding to cut vertices are placeholders
    * All nodes corresponding to vertices in non_reachable partition are recomputed in B
      (except the nodes corresponding to cut vertices, whose are placeholders).
    * Remove the old placeholders that are now re-computed


    We want to modify fused_graph such that:
    * If A has additional outputs, we also need to add getitem nodes in the fused graph such that B can access them.
    * If A originally has a single output, but now has multiple outputs,
      we also need to change the users of A in fused_graph now to use the new getitem(A, 0) node.
    * The input args of node B in fused_graph need to be changed to match the new placeholders of B

    Args:
        node_pair: a pair of nodes in the fused graph.
        fused_graph: the fused graph.
        name_to_node: a dictionary from node name to node.
        ``name_to_node`` is a mapping from name to nodes in fused graph. It is used to modify the args of
        nodes. In general the name nodes corresponds to each other, except one case.
        e.g. if a node "fused_0" orignally has a single output, but after rematerialization it has multiple outputs,
        name_to_node["fused_0"] will be changed to map to the new getitem(fused_0, 0).

        partition: a tuple of two sets. The first set is the reachable nodes in the fused graph, and the second set
        is the non-reachable nodes in the fused graph.

        cut_nodes: a set of nodes that cut across the partition
    """
    reachable, non_reachable = partition
    module_origin = getattr(fused_graph, node_pair[0].name)
    module_dest = getattr(fused_graph, node_pair[1].name)

    # map from placeholder name in modules_dest to node_pair[1]'s args
    placeholder_map = get_name_to_args_map(node_pair[1], module_dest)

    name_to_node_origin = {node.name: node for node in module_origin.graph.nodes}
    name_to_node_dest = {node.name: node for node in module_dest.graph.nodes}

    # modify the outputs of module_origin to contain new outputs
    module_origin_new_outputs = {name_to_node_origin[name] for name in cut_nodes}
    add_new_outputs(node_pair, fused_graph, name_to_node, module_origin, module_origin_new_outputs)

    first_node_dest = None
    for node in module_dest.graph.nodes:
        first_node_dest = node
        break

    env = {}  # map from node in origin to node in dest
    # create new placeholders in module_dest
    for node_name in cut_nodes:
        node = name_to_node_origin[node_name]
        if node_name in name_to_node_dest:
            # already has a placeholder for it in dest
            env[node] = name_to_node_dest[node_name]
            continue
        with module_dest.graph.inserting_before(first_node_dest):
            new_node = module_dest.graph.placeholder(node.name, type_expr=node.type)
            new_node.meta = copy.copy(node.meta)
            env[node] = new_node

    # copy internal nodes
    node_to_copy = get_node_to_copy(non_reachable, cut_nodes)
    node_to_copy = order_topologically(node_to_copy, module_origin)
    for node_name in node_to_copy:
        node = name_to_node_origin[node_name]
        with module_dest.graph.inserting_before(first_node_dest):
            new_node = module_dest.graph.node_copy(node, lambda x: env[x])
            new_node.name = node.name  # use the same name such that node can be referenced back to original graph
            env[node] = new_node
            # change the args of nodes in dest to use the new node
            # e.g. a placeholder is now a node to recompute
            if node.name in name_to_node_dest:
                name_to_node_dest[node.name].replace_all_uses_with(new_node)

    # erase unused placeholder nodes and record current active placeholders
    active_placeholders = []
    for node in module_dest.graph.nodes:
        if node.op == "placeholder":
            if len(node.users) == 0:
                module_dest.graph.erase_node(node)
            else:
                active_placeholders.append(node.name)

    legalize_graph(module_dest)
    module_dest.graph.eliminate_dead_code()
    module_dest.graph.lint()

    # change the args of node_pair[1] to match the new placeholders of module_dest
    origin_placeholder_map = get_name_to_args_map(node_pair[0], module_origin)
    placeholder_map.update(origin_placeholder_map)
    for node in module_origin.graph.nodes:
        if node.op == "output":
            user_name_to_user_map = get_user_name_to_user_map(node, node_pair[0])
            placeholder_map.update(user_name_to_user_map)

    new_args = []
    for name in active_placeholders:
        if name in name_to_node:  # name is a node in fused graph
            new_args.append(name_to_node[name])
        else:  # name is a placeholder in origin's module or dest's module or a newly added input
            new_args.append(placeholder_map[name])
    node_pair[1].args = tuple(new_args)
    fused_graph.recompile()

    # fused_graph.graph.eliminate_dead_code()
    # fused_graph.graph.lint()
    module_dest.recompile()

    # remove the unused output to write less
    remove_unused_output(fused_graph, module_origin)


def find_min_cut(node_pair: Tuple[fx.Node, fx.Node], node_users_map: Dict[str, Set[fx.Node]], fused_graph: fx.GraphModule):
    """
    Use a min-cut algorithm to determine whether rematerialization between the ```node_pair```
    is needed and which nodes should be rematerialized.
    This algorithm minimizes the memory reading/writing cost of computing the outputs of these two fusion groups.
    The mincut value is the cost of reading/writing between the two fusion groups.

    Args:
        node_pair (Tuple[fx.Node, fx.Node]): a pair of nodes in the fusion graph. We want to find the mincut between them.
            The first node is the origin node and the second node is the destination node. They should be "touching",
            meaning the destination node should use the outputs of the origin node, either directly or through a getitem node.
        node_users_map (Dict[str, Set[fx.Node]]): map from a node's name its users in the original graph G
        fused_graph (fx.GraphModule): the fused graph (of some FX graph G) where each fusion group is a submodule
    """

    try:
        import networkx as nx
    except ImportError:
        raise RuntimeError("Need networkx installed to perform smart recomputation heuristics")
    nx_graph = nx.DiGraph()
    node_origin = node_pair[0]
    node_dest = node_pair[1]
    module_origin = getattr(fused_graph, node_origin.name)
    module_dest = getattr(fused_graph, node_dest.name)

    dest_placeholder_names = {node.name for node in module_dest.graph.nodes if node.op == "placeholder"}
    # used to check if a node has users in dest.
    # The user node in the original graph has the same name as the call_func nodes in dest.
    dest_node_names = {node.name for node in module_dest.graph.nodes if node.op != "placeholder" and node.op != "output"}  # noqa: E501
    orig_node_names = {node.name for node in module_origin.graph.nodes if node.op != "placeholder" and node.op != "output"}  # noqa: E501

    # track the users of each node in traced_graph
    getitem_users = {}
    for node in module_origin.graph.nodes:
        if node.op == "output":
            output_args = get_output_node_args(node)
    loc = 0
    for user in node_origin.users:
        # can only do this for getitem users. might have a single add node that have two users
        if user.target != operator.getitem:
            break
        if isinstance(output_args[loc], torch.fx.node.Node):
            user_name = output_args[loc].name
            getitem_users[user_name] = user.name  # add new arg to dest placeholder map
        loc += 1

    def get_capacity(node):
        # if rematerialize an internal node, need to read and write
        # might not need to add the write cost, because it might be read by other
        # might not need to add the read cost, if already reading it - no need the cost
        user_names_set = set({n.name for n in node_users_map[node.name]})
        user_names_outside_set = user_names_set.difference(orig_node_names)
        write_cost = 0  # cost for both read and write because only dest_module is using it
        if weight and user_names_outside_set.issubset(set(dest_node_names)):
            write_cost = weight

        read_cost = weight

        capacity = write_cost + read_cost
        return capacity

    for node in module_origin.graph.nodes:
        if node.op == 'output':
            continue

        weight = get_weight(node)

        if ban_recomputation(node):
            nx_graph.add_edge("source", node.name + "_out", capacity=math.inf)

        # some ops like cuda_batch_norm return tuples, and they cannot be returned as output
        # because torch.jit.script does not accept tuples.
        # so we need to change the capacity between _in and _out of these tuple nodes to inf
        # Instead, we need to return getitem nodes, and these getitems might already be in the graph,
        for user in node.users:
            if user.target == operator.getitem:
                weight = math.inf
            break

        if node.op == 'placeholder':
            capacity = weight
            nx_graph.add_edge("source", node.name + "_in", capacity=math.inf)
        elif node.op == 'call_function':
            capacity = get_capacity(node)

        if (node.name in dest_placeholder_names or
           (node.name in getitem_users and getitem_users[node.name] in dest_placeholder_names)):
            nx_graph.add_edge(node.name + "_out", 'sink', capacity=capacity)

        nx_graph.add_edge(node.name + "_in", node.name + "_out", capacity=capacity)
        for user in node.users:
            if user.op != "output":
                nx_graph.add_edge(node.name + "_out", user.name + "_in", capacity=math.inf)

    cut_value, partition = nx.minimum_cut(nx_graph, "source", "sink")

    cut_at_sink = 0
    for e in nx_graph.edges.data():
        if e[1] == "sink":
            cut_at_sink += e[2]["capacity"]

    local_memory_reduced = cut_at_sink - cut_value
    cut_nodes = get_cut_nodes_from_partition(partition, nx_graph)

    return partition, cut_nodes, local_memory_reduced


def get_fused_graph(traced_graph):
    supported_ops = NvFuserOperatorSupport()
    partitioner = CapabilityBasedPartitioner(traced_graph, supported_ops)
    fused_graph = partitioner.partition_and_fuse()
    return fused_graph


def rematerialize_fused_graph(fused_graph: fx.GraphModule, node_users_map: Dict[str, Set[fx.Node]]):
    """
    Modify the fused graph to rematerialize the nodes.
    We find all pairs of fusion groups (A, B) that are “touching”, meaning fusion group A's output
    is directly used by fusion group B.

    The for each pair of nodes, we use a min-cut algorithm to determine whether rematerialization
    is needed and which nodes should be rematerialized. This algorithm minimizes the memory reading/writing
    cost of computing the outputs of these two fusion groups. Then we modify the fusion groups to
    rematerialize the nodes based on the min-cut partition given.

    This is a heuristic algorithm because even though the rematerialization decision between each pair of
    touching fusion groups is optimal, the decision might not be the optimal for the whole graph.

    Args:
        fused_graph: the fused graph (of some FX graph G) where each fusion group is a submodule
        node_users_map (Dict[str, Set[fx.Node]]): map from a node's name its users in the original graph G
    """
    global num_group_remat, num_node_pairs, memory_reduced
    name_to_node = {node.name: node for node in fused_graph.graph.nodes}

    fused_node_pairs = get_fused_node_pairs(fused_graph)
    num_node_pairs = len(fused_node_pairs)
    for node_pair in fused_node_pairs:
        partition, cut_nodes, local_memory_reduced = find_min_cut(node_pair, node_users_map, fused_graph)
        if local_memory_reduced > 0:
            num_group_remat += 1
            memory_reduced += local_memory_reduced
            copy_nodes(node_pair, fused_graph, name_to_node, partition, cut_nodes)
    return fused_graph


def rematerialize(traced_graph: fx.GraphModule):
    """
    Modify traced_graph to a graph with rematerialization.
    The returned graph has a submodule fused_* for each fusion group.

    It first obtains a fused graph, then rematerialize the fused graph.
    """
    traced_graph.graph.eliminate_dead_code()
    traced_graph.recompile()
    node_users_map = {node.name: set(node.users.keys()) for node in traced_graph.graph.nodes}

    fused_graph = get_fused_graph(traced_graph)
    return rematerialize_fused_graph(fused_graph, node_users_map)


def rematerialize_stat(traced_graph: fx.GraphModule, stat: Dict[str, int]):
    """
    Modify traced_graph to a graph with rematerialization.
    The returned graph has a submodule fused_* for each fusion group.

    It first obtains a fused graph, then rematerialize the fused graph.

    ``stat`` is a dictionary and the stats of rematerialization will be stored in it.
    """
    global num_group_remat, memory_reduced, num_node_pairs
    num_group_remat = 0
    memory_reduced = 0
    traced_graph.graph.eliminate_dead_code()
    traced_graph.recompile()
    node_users_map = {node.name: set(node.users.keys()) for node in traced_graph.graph.nodes}

    fused_graph = get_fused_graph(traced_graph)
    fused_graph = rematerialize_fused_graph(fused_graph, node_users_map)

    stat["num_group_remat"] = num_group_remat
    stat["memory_reduced"] = memory_reduced
    stat["num_node_pairs"] = num_node_pairs
    return fused_graph
