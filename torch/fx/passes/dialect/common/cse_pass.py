from typing import Dict, Tuple, Any

import torch
import torch.fx as fx
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils._pytree import tree_flatten

aten = torch.ops.aten


def get_aten_target(node):
    if hasattr(node.target, 'overloadpacket'):
        return node.target.overloadpacket
    return node.target

# random ops are banned from CSE
rand_ops = [aten.dropout, aten._fused_dropout, aten._standard_gamma,
            aten.bernoulli, aten.multinomial, aten.native_dropout,
            aten.normal, aten.poisson, aten.binomial, aten.rrelu,
            aten.rand_like, aten.rand, aten.randint, aten.randn, aten.randperm]


class CSEPass(PassBase):

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        """
        Return a new copy of torch.fx.GraphModule with CSE applied to the input graph

        Example usage:

        from torch.fx.experimental.proxy_tensor import make_fx
        def f(a):
            b = a * a
            c = a * a
            return b+c

        p = CSEPass()
        traced_graph = make_fx(f)(torch.tensor(1))
        print(traced_graph)
        result = p(traced_graph)
        print(result.graph_module)
        """
        modified = False
        new_graph = fx.Graph()
        env: Dict[fx.node.Node, fx.node.Node] = {}  # map from node in the old graph to node in the new graph
        hash_env: Dict[Tuple[torch._ops.OpOverload, int], fx.node.Node] = {}  # map from hash to a node in the new graph
        token_map: Dict[Tuple[torch._ops.OpOverload, int], Dict[str, Any]] = {}  # map from hash to token
        for n in graph_module.graph.nodes:
            # The placeholder, output, and get_attr nodes are copied to the new grpah without change
            # do not CSE away random operations
            if n.op == 'placeholder' or n.op == 'output' or n.op == 'get_attr' or get_aten_target(n) in rand_ops:
                new_node = new_graph.node_copy(n, lambda x: env[x])
                env[n] = new_node
            else:  # n.op == 'call_function', should never see n.op == 'call_module' or 'call_method'
                # substitute args and kwargs memebrs to their mapping in env if exists
                # specs can be used to reconstruct nested list/dictionaries
                def substitute(arg_list):
                    arg_list, spec = tree_flatten(arg_list)
                    for i in range(len(arg_list)):
                        v = arg_list[i]
                        if isinstance(v, torch.fx.node.Node) and v in env:
                            arg_list[i] = env[v]
                    return tuple(arg_list), spec
                args, args_spec = substitute(n.args)
                kwargs, kwargs_spec = substitute(n.kwargs)

                # each token corresponds to a unique node
                # nodes with the same token can be substituted
                token = {"target": n.target, "args": args, "args_spec": args_spec,
                         "kwargs": kwargs, "kwargs_spec": kwargs_spec}

                # hash substituted args to a number, do not hash specs because specs are not hashable
                hash_arg = hash((args, kwargs))
                hash_val = (n.target, hash_arg)

                # check if a node has a substitute and can be eliminated
                hash_val_in_hash_env = hash_val in hash_env
                if hash_val_in_hash_env and token_map[hash_val] == token:
                    modified = True  # substition happens and the graph is modified
                    env[n] = hash_env[hash_val]
                    continue

                new_node = new_graph.node_copy(n, lambda x: env[x])
                env[n] = new_node
                if not hash_val_in_hash_env:
                    hash_env[hash_val] = new_node
                    token_map[hash_val] = token

        csed_gm = fx.GraphModule(graph_module, new_graph)
        return PassResult(csed_gm, modified)
