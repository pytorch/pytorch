# mypy: ignore-errors


import torch
import torch.fx as fx
from torch.utils._pytree import tree_flatten
from torch.utils import _pytree as pytree

aten = torch.ops.aten


def get_aten_target(node):
    if hasattr(node.target, 'overloadpacket'):
        return node.target.overloadpacket
    return node.target


rand_ops = [aten.dropout, aten._fused_dropout, aten._standard_gamma,
            aten.bernoulli, aten.multinomial, aten.native_dropout,
            aten.normal, aten.poisson, aten.binomial, aten.rrelu,
            aten.rand_like, aten.rand, aten.randint, aten.randn, aten.randperm]


# return a new copy of torch.fx.graph.Graph with CSE applied to the input graph
def fx_graph_cse(fx_g: torch.fx.graph.Graph):
    new_graph = fx.Graph()
    env = {}  # map from node in the old graph to node in the new graph
    hash_env = {}  # map from hash to a node in the new graph
    token_map = {}  # map from hash to token
    for n in fx_g.nodes:
        # The placeholder, output, and get_attr nodes are copied to the new graph without change
        # do not CSE away random operations
        if n.op == 'placeholder' or n.op == 'output' or n.op == 'get_attr' or get_aten_target(n) in rand_ops:
            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
        else:  # n.op == 'call_function', should never see n.op == 'call_module' or 'call_method'
            # substitute args and kwargs members to their mapping in env if exists
            # specs can be used to reconstruct nested list/dictionaries
            def substitute(arg_list):
                arg_list, spec = tree_flatten(arg_list)
                for i in range(len(arg_list)):
                    v = arg_list[i]
                    if isinstance(v, torch.fx.node.Node) and v in env:
                        arg_list[i] = env[v]
                    if isinstance(v, (torch.SymBool, torch.SymInt, torch.SymFloat)):
                        arg_list[i] = v.node
                return tuple(arg_list), spec
            args, args_spec = substitute(n.args)
            kwargs, kwargs_spec = substitute(n.kwargs)

            # each token corresponds to a unique node
            # nodes with the same token can be substituted
            token = {"target": n.target, "args": args, "args_spec": args_spec,
                     "kwargs": kwargs, "kwargs_spec": kwargs_spec}

            # hash substituted args to a number, do not hash specs because specs are not hashable
            # We need to add type into hash to avoid situations like:
            # hash((primals_2, 1.0)) == hash((primals_2, 1))
            hash_arg = hash((tuple((a, type(a)) for a in args), tuple((a, type(a)) for a in kwargs)))
            hash_val = (n.target, hash_arg)

            # check if a node has a substitute and can be eliminated
            hash_val_in_hash_env = hash_val in hash_env
            if hash_val_in_hash_env and token_map[hash_val] == token:
                env[n] = hash_env[hash_val]
                continue

            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
            if not hash_val_in_hash_env:
                hash_env[hash_val] = new_node
                token_map[hash_val] = token

    return new_graph


def strip_overloads(gm):
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.

    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            node.target = node.target.overloadpacket
    gm.recompile()


def get_placeholders(graph):
    return list(filter(lambda x: x.op == 'placeholder', graph.nodes))

def get_outputs(graph):
    for node in graph.nodes:
        if node.op == 'output':
            return pytree.tree_leaves(node.args[0])
    raise AssertionError("No output node found")
