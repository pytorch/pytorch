from typing import Dict, List, Optional, Tuple

import torch
import torch.fx

from .graph_signature import ExportGraphSignature, InputKind, InputSpec


def _reorder_placeholder_same_as_original_ep_pass(
    gm: torch.fx.GraphModule,
    new_graph_signature: ExportGraphSignature,
    param_buffer_original_index_to_current_idx: Dict[int, int],
) -> Tuple[torch.fx.GraphModule, ExportGraphSignature]:
    """
    When we run_decomp, we first call ep.module and then retrace.
    As a result, the order of placeholders can be different because adding params into module is
    not guaranteed to be same.
    """
    param_buffer_name_to_idx = {}
    idx_to_param_buffer_name = {}
    for ix, name in enumerate(
        (*new_graph_signature.parameters, *new_graph_signature.buffers)
    ):
        param_buffer_name_to_idx[name] = ix
        idx_to_param_buffer_name[ix] = name

    node_to_metadata = {}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node_to_metadata[node.name] = node.meta
        else:
            break

    param_buffer_name_to_graph_input_name = {}
    for k, v in {
        **new_graph_signature.inputs_to_parameters,
        **new_graph_signature.inputs_to_buffers,
    }.items():
        param_buffer_name_to_graph_input_name[v] = k

    name_to_node = {}
    corrected_name_to_old_name = {}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            old_name = node.name
            if old_name in new_graph_signature.inputs_to_parameters:
                param_name = new_graph_signature.inputs_to_parameters[old_name]
                if param_name in param_buffer_name_to_idx:
                    current_idx = param_buffer_name_to_idx[param_name]
                    if current_idx in param_buffer_original_index_to_current_idx:
                        actual_idx = param_buffer_original_index_to_current_idx[
                            current_idx
                        ]
                        corrected_param_name = idx_to_param_buffer_name[actual_idx]
                        correct_input_node_name = param_buffer_name_to_graph_input_name[
                            corrected_param_name
                        ]
                        node.name = node.target = correct_input_node_name
                        node.meta = node_to_metadata[correct_input_node_name]
                        corrected_name_to_old_name[correct_input_node_name] = old_name
            if old_name in new_graph_signature.inputs_to_buffers:
                param_name = new_graph_signature.inputs_to_buffers[old_name]
                if param_name in param_buffer_name_to_idx:
                    current_idx = param_buffer_name_to_idx[param_name]
                    if current_idx in param_buffer_original_index_to_current_idx:
                        actual_idx = param_buffer_original_index_to_current_idx[
                            current_idx
                        ]
                        corrected_param_name = idx_to_param_buffer_name[actual_idx]
                        correct_input_node_name = param_buffer_name_to_graph_input_name[
                            corrected_param_name
                        ]
                        node.name = node.target = correct_input_node_name
                        node.meta = node_to_metadata[correct_input_node_name]
                        corrected_name_to_old_name[correct_input_node_name] = old_name
            name_to_node[node.name] = node

        else:
            new_args = []
            for arg in node.args:
                if (
                    isinstance(arg, torch.fx.Node)
                    and arg.name in corrected_name_to_old_name
                ):
                    new_args.append(name_to_node[corrected_name_to_old_name[arg.name]])
                else:
                    new_args.append(arg)
            node.args = tuple(new_args)

    new_input_specs: List[Optional[InputSpec]] = [
        None for _ in range(len(new_graph_signature.input_specs))
    ]
    param_buffer_spec_name_to_spec = {}
    for ix, input_spec in enumerate(new_graph_signature.input_specs):
        if input_spec.kind == InputKind.PARAMETER:
            param_buffer_spec_name_to_spec[
                new_graph_signature.inputs_to_parameters[input_spec.arg.name]
            ] = input_spec
        if input_spec.kind == InputKind.BUFFER:
            param_buffer_spec_name_to_spec[
                new_graph_signature.inputs_to_buffers[input_spec.arg.name]
            ] = input_spec

    for ix, input_spec in enumerate(new_graph_signature.input_specs):
        if input_spec.kind not in [InputKind.PARAMETER, InputKind.BUFFER]:
            new_input_specs[ix] = input_spec
        if input_spec.kind == InputKind.PARAMETER:
            name = input_spec.arg.name
            param_name = new_graph_signature.inputs_to_parameters[name]
            if param_name in param_buffer_name_to_idx:
                current_idx = param_buffer_name_to_idx[param_name]
                if current_idx in param_buffer_original_index_to_current_idx:
                    original_idx = param_buffer_original_index_to_current_idx[
                        current_idx
                    ]
                    corrected_param_name = idx_to_param_buffer_name[original_idx]
                    new_input_specs[ix] = param_buffer_spec_name_to_spec[
                        corrected_param_name
                    ]
        if input_spec.kind == InputKind.BUFFER:
            name = input_spec.arg.name
            param_name = new_graph_signature.inputs_to_buffers[name]
            if param_name in param_buffer_name_to_idx:
                current_idx = param_buffer_name_to_idx[param_name]
                if current_idx in param_buffer_original_index_to_current_idx:
                    original_idx = param_buffer_original_index_to_current_idx[
                        current_idx
                    ]
                    corrected_param_name = idx_to_param_buffer_name[original_idx]
                    new_input_specs[ix] = param_buffer_spec_name_to_spec[
                        corrected_param_name
                    ]

        if new_input_specs[ix] is None:
            new_input_specs[ix] = input_spec

    assert all(x is not None for x in new_input_specs)

    gm.recompile()
    new_graph_signature = ExportGraphSignature(
        input_specs=new_input_specs, output_specs=new_graph_signature.output_specs  # type: ignore[arg-type]
    )
    return gm, new_graph_signature
