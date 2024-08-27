import copy
from typing import Dict, Tuple

import torch
import torch.fx

from .graph_signature import ExportGraphSignature, InputKind


def _reorder_placeholder_same_as_original_ep_pass(
    gm: torch.fx.GraphModule,
    new_graph_signature: ExportGraphSignature,
    old_gm: torch.fx.GraphModule,
    old_graph_signature: ExportGraphSignature,
) -> Tuple[torch.fx.GraphModule, ExportGraphSignature]:
    """
    When we run_decomp, we first call ep.module and then retrace.
    As a result, the order of placeholders can be different because adding params into module is
    not guaranteed to be same.
    """
    old_gm_node_name_to_node: Dict[str, torch.fx.Node] = {}
    for node in old_gm.graph.nodes:
        if node.op == "placeholder":
            old_gm_node_name_to_node[node.name] = node
        else:
            break

    name_to_node_in_new_graph: Dict[str, torch.fx.Node] = {}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            name_to_node_in_new_graph[node.name] = node
        else:
            break

    new_graph = torch.fx.Graph()
    val_map: Dict[torch.fx.Node, torch.fx.Node] = {}
    for name in old_gm_node_name_to_node:
        node = new_graph.placeholder(name)
        node.meta = copy.copy(old_gm_node_name_to_node[name].meta)
        cor_node = name_to_node_in_new_graph[name]
        val_map[cor_node] = node

    output = new_graph.graph_copy(gm.graph, val_map)
    new_graph.output(output)
    gm.graph = new_graph

    non_user_inp_count = len(
        [x for x in new_graph_signature.input_specs if x.kind != InputKind.USER_INPUT]
    )
    # We only need to adjust the input specs for the non user inputs. For user inputs, it is bit annoying
    # because our placeholder prettify pass adds kwargs_ to the kwarg input names. But when we run ep.module()
    # there is no concept of kwargs anymore. We could fix it by manually adding kwargs to newly created ep after
    # decomp, but that seems too overkill.
    new_graph_signature = ExportGraphSignature(
        input_specs=old_graph_signature.input_specs[:non_user_inp_count] + new_graph_signature.input_specs[non_user_inp_count:],  # type: ignore[arg-type]
        output_specs=new_graph_signature.output_specs,  # type: ignore[arg-type]
    )
    return gm, new_graph_signature
