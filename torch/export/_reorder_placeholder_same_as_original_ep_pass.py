from typing import Dict, List, Tuple

import torch
import torch.fx
import torch.utils._pytree as pytree

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
    node_to_metadata = {}
    original_order_to_node_name: List[str] = []
    for node in old_gm.graph.nodes:
        if node.op == "placeholder":
            node_to_metadata[node.name] = node.meta
            original_order_to_node_name.append(node.name)
        else:
            break

    name_to_node: Dict[str, torch.fx.Node] = {}
    corrected_name_to_old_name: Dict[str, str] = {}
    count = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if new_graph_signature.input_specs[count].kind != InputKind.USER_INPUT:
                cur_name = node.name
                correct_name = original_order_to_node_name[len(name_to_node)]
                node.name = node.target = correct_name
                node.meta = node_to_metadata[correct_name]
                corrected_name_to_old_name[correct_name] = cur_name
                name_to_node[cur_name] = node
            count += 1
        else:

            def replace_with_correct_node(x: torch.fx.Node) -> torch.fx.Node:
                if x.name in corrected_name_to_old_name:
                    return name_to_node[corrected_name_to_old_name[x.name]]
                return x

            new_args = pytree.tree_map_only(
                torch.fx.Node, lambda x: replace_with_correct_node(x), node.args
            )
            new_kwargs = pytree.tree_map_only(
                torch.fx.Node, lambda x: replace_with_correct_node(x), node.kwargs
            )
            node.args = new_args
            node.kwargs = new_kwargs

    gm.recompile()
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
