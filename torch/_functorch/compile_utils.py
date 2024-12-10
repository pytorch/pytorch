


from typing import Callable

import torch
import torch.fx as fx
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten


aten = torch.ops.aten


def get_aten_target(node: fx.Node) -> Callable:
    if hasattr(node.target, "overloadpacket"):
        return node.target.overloadpacket
    return node.target


rand_ops = [
    aten.dropout,
    aten._fused_dropout,
    aten._standard_gamma,
    aten.bernoulli,
    aten.multinomial,
    aten.native_dropout,
    aten.normal,
    aten.poisson,
    aten.binomial,
    aten.rrelu,
    aten.rand_like,
    aten.rand,
    aten.randint,
    aten.randn,
    aten.randperm,
]


# return a new copy of torch.fx.graph.Graph with CSE applied to the input graph
def fx_graph_cse(fx_g: torch.fx.graph.Graph):
    new_graph = fx.Graph()
    env = {}  # map from node in the old graph to node in the new graph
    hash_env = {}  # map from hash to a node in the new graph
    token_map = {}  # map from hash to token

    from torch._inductor.pattern_matcher import (
        compute_mutation_region_ids,
        same_mutation_regions,
    )

    compute_mutation_region_ids(fx_g)  # type: ignore[arg-type]

    # Make a set of separate storages returned from the output, which will be preserved
    # when pruning.  This prevents us from deduplicating returned tensors which have
    # experienced identical operations, but are separate data structures in eager mode.
    output_node: fx.Node = list(fx_g.nodes)[-1]
    assert output_node.op == "output"

    def checkable_node(node: fx.Node) -> bool:
        """We can evaluate only nodes that represent tensors with defined storage."""
        if "val" not in node.meta or not isinstance(node.meta["val"], torch.Tensor):
            return False

        try:
            node.meta["val"].untyped_storage()
        except NotImplementedError:
            return False

        return True

    output_storages = {
        StorageWeakRef(n.meta["val"].untyped_storage())
        for n in output_node.all_input_nodes
        if checkable_node(n)
    }
    nodes_that_alias_outputs = {
        n
        for n in fx_g.nodes
        if checkable_node(n)
        and StorageWeakRef(n.meta["val"].untyped_storage()) in output_storages
    }

    for n in fx_g.nodes:
        # The placeholder, output, and get_attr nodes are copied to the new graph without change
        # do not CSE away random operations
        if (
            n.op == "placeholder"
            or n.op == "output"
            or n.op == "get_attr"
            or get_aten_target(n) in rand_ops
            # aten.empty is non-deterministic, so don't CSE it.
            # Also, aten.empty is almost always fusible into its consumer,
            # so it's not worth CSEing.
            or get_aten_target(n) is aten.empty
            or n in nodes_that_alias_outputs
        ):
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
            token = {
                "target": n.target,
                "args": args,
                "args_spec": args_spec,
                "kwargs": kwargs,
                "kwargs_spec": kwargs_spec,
            }

            # hash substituted args to a number, do not hash specs because specs are not hashable
            # We need to add type into hash to avoid situations like:
            # hash((primals_2, 1.0)) == hash((primals_2, 1))
            hash_arg = hash(
                (tuple((a, type(a)) for a in args), tuple((a, type(a)) for a in kwargs))
            )
            hash_val = (n.target, hash_arg)

            # check if a node has a substitute and can be eliminated
            hash_val_in_hash_env = hash_val in hash_env
            overwrite_due_to_mutation = False
            if hash_val_in_hash_env and token_map[hash_val] == token:
                duplicate_n_prev = hash_env[hash_val]
                if same_mutation_regions(n, duplicate_n_prev):
                    env[n] = duplicate_n_prev
                    continue
                else:
                    # any futures duplicates should replace with n, not duplicate_n_prev
                    overwrite_due_to_mutation = True

            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
            if overwrite_due_to_mutation or not hash_val_in_hash_env:
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
    return graph.find_nodes(op="placeholder")


def get_outputs(graph):
    for node in graph.find_nodes(op="output"):
        return pytree.tree_leaves(node.args[0])
    raise AssertionError("No output node found")
