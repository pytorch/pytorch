import copy
from typing import Dict, List, Tuple

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
    targets = {
        spec.target: ix
        for ix, spec in enumerate(old_graph_signature.input_specs)
        if spec.kind != InputKind.USER_INPUT
    }

    count = 0
    nodes_in_correct_order = [None for _ in targets]
    user_inputs: List[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if new_graph_signature.input_specs[count].kind != InputKind.USER_INPUT:
                target = new_graph_signature.input_specs[count].target
                assert target in targets
                nodes_in_correct_order[targets[target]] = node
            else:
                user_inputs.append(node)
            count += 1
        else:
            break

    new_graph = torch.fx.Graph()
    val_map: Dict[torch.fx.Node, torch.fx.Node] = {}
    for node in nodes_in_correct_order:
        assert isinstance(node, torch.fx.Node)
        new_node = new_graph.placeholder(node.name)
        new_node.meta = copy.copy(node.meta)
        val_map[node] = new_node

    # This is odd use case where when the placeholder name conflicts with
    # builtin type name, fx.graph tries to mangle the name
    for node in user_inputs:
        new_node = new_graph.placeholder(node.name)
        new_node.name = node.name
        new_node.meta = copy.copy(node.meta)
        val_map[node] = new_node

    output = new_graph.graph_copy(gm.graph, val_map)
    new_graph.output(output)
    gm.graph = new_graph

    target_name_to_input_specs = {
        spec.target: spec
        for spec in new_graph_signature.input_specs
        if spec.kind != InputKind.USER_INPUT
    }

    new_input_specs_for_non_user_inputs = copy.copy(
        new_graph_signature.input_specs[: len(target_name_to_input_specs)]
    )
    for target, ix in targets.items():
        spec = target_name_to_input_specs[target]
        new_input_specs_for_non_user_inputs[ix] = spec

    # We only need to adjust the input specs for the non user inputs. For user inputs, it is bit annoying
    # because our placeholder prettify pass adds kwargs_ to the kwarg input names. But when we run ep.module()
    # there is no concept of kwargs anymore. We could fix it by manually adding kwargs to newly created ep after
    # decomp, but that seems too overkill.
    new_graph_signature = ExportGraphSignature(
        input_specs=new_input_specs_for_non_user_inputs + new_graph_signature.input_specs[len(target_name_to_input_specs) :],  # type: ignore[arg-type]
        output_specs=new_graph_signature.output_specs,  # type: ignore[arg-type]
    )
    return gm, new_graph_signature
