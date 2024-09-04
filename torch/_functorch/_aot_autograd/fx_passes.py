# mypy: allow-untyped-defs
import itertools
import logging
import typing
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Union

import torch

def remove_fsdp2_unsharded_param_graph_input_usage(graph: torch.fx.Graph):
    # log.warn("remove_fsdp2_unsharded_param_graph_input_usage is called!")
    # Check condition 1: fullgraph=True  # TODO(yf225): find a way to check this
    # To check full-graph, either check top-level config value or (maybe) check that the two `unsharded_param.resize_`s cancel out so we know it's full graph

    node_list = list(graph.nodes)

    # Find all unsharded params and their corresponding graph intermediates.
    unsharded_param_to_fsdp_copy_node_idxes = defaultdict(list)
    for idx, node in enumerate(node_list):
        if node.op == "call_function" and node.target == torch.ops.fsdp.copy_.default:
            fsdp_copy_node = node
            unsharded_param = node.args[0]
            assert unsharded_param.op == "placeholder", "Assumed all FSDP2 `unsharded_param`s to be graph input, but it's not true!"
            unsharded_param_to_fsdp_copy_node_idxes[unsharded_param].append(idx)
    
    # Check no user mutation on any unsharded_param
    def is_allowed_mutation(node):
        return (node.target == torch.ops.fsdp.copy_.default or 
                node.target == torch.ops.inductor.resize_storage_bytes_.default)

    for node in node_list:
        if node.op == "call_function" and isinstance(node.target, torch._ops.OpOverload) and node.target._schema.is_mutable and not is_allowed_mutation(node) and any(arg in unsharded_param_to_graph_intermediate for arg in node.args):
            raise RuntimeError("User mutation on FSDP2 unsharded param is not allowed when Traceable FSDP2 is used")

    # For each `fsdp.copy_(unsharded_param, Y)`, replace downstream usage of `unsharded_param` with `Y`.
    #
    # NOTE: Because of "layer reuse" use case, there could be multiple `fsdp.copy_` to the same `unsharded_param` graph input.
    # e.g.
    # ```
    #     fsdp_copy_1 = fsdp.copy_(unsharded_param_1, Y1)
    #     ... (use of unsharded_param_1)                     -> Subgraph 1
    #     fsdp_copy_2 = fsdp.copy_(unsharded_param_1, Y2)
    #     ... (use of unsharded_param_1)                     -> Subgraph 2
    #     fsdp_copy_3 = fsdp.copy_(unsharded_param_1, Y3)
    #     ... (use of unsharded_param_1)                     -> Subgraph 3
    # ```
    # We must do the replacement only within each subgraph.
    replacement_nodes = set()
    for unsharded_param, fsdp_copy_node_idxes in unsharded_param_to_fsdp_copy_node_idxes.items():
        for i, fsdp_copy_node_idx in enumerate(fsdp_copy_node_idxes):
            fsdp_copy_node = node_list[fsdp_copy_node_idx]
            assert fsdp_copy_node.args[0] is unsharded_param
            _, replacement = fsdp_copy_node.args
            replacement_nodes.add(replacement)
            # subgraph_start_idx is inclusive
            subgraph_start_idx = fsdp_copy_node_idx
            # subgraph_end_idx is exclusive (also intentionally don't replace args in return op)
            subgraph_end_idx = fsdp_copy_node_idxes[i+1] if i < len(fsdp_copy_node_idxes) - 1 else len(node_list) - 1
            subgraph = node_list[subgraph_start_idx:subgraph_end_idx]
            for node in subgraph:
                if node.op == "call_function" and unsharded_param in node.args:  # TODO(yf225): implement replacement in kwargs
                    new_args = tuple(replacement if arg is unsharded_param else arg for arg in node.args)
                    node.args = new_args
    
    # Delete `fsdp.copy_(unsharded_param, Y)` nodes
    for unsharded_param, fsdp_copy_node_idxes in unsharded_param_to_fsdp_copy_node_idxes.items():
        for i, fsdp_copy_node_idx in enumerate(fsdp_copy_node_idxes):
            fsdp_copy_node = node_list[fsdp_copy_node_idx]
            graph.erase_node(fsdp_copy_node)
    
    # Delete resize nodes, including:
    # 1. `resize_storage_bytes_(unsharded_param, ...)` nodes
    # 2. `resize_storage_bytes_(Y, 0)` nodes
    for node in node_list:
        if node.op == "call_function" and node.target == torch.ops.inductor.resize_storage_bytes_.default and (node.args[0] in unsharded_param_to_fsdp_copy_node_idxes or node.args[0] in replacement_nodes):
            graph.erase_node(node)
