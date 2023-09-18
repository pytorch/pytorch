import copy

import torch
from torch._export import ExportedProgram
from torch._guards import detect_fake_mode


def lift_constant_tensor_pass(ep: ExportedProgram) -> ExportedProgram:
    if len([node for node in ep.graph.nodes if node.op == "placeholder"]) == 0:
        return ep

    graph_signature = ep.graph_signature
    inputs_to_buffers = graph_signature.inputs_to_buffers
    buffers = graph_signature.buffers

    fake_mode = detect_fake_mode(
        tuple(node.meta["val"] for node in ep.graph.nodes if node.op == "placeholder")
    )
    assert fake_mode is not None

    first_user_input = None
    for node in ep.graph.nodes:
        if node.op == "placeholder" and node.name in graph_signature.user_inputs:
            first_user_input = node
            break

    for node in ep.graph.nodes:
        if node.op == "get_attr":
            constant_tensor = getattr(ep.graph_module, node.target)
            if not isinstance(constant_tensor, torch.Tensor):
                continue

            constant_tensor_fqn = node.target

            with ep.graph.inserting_before(first_user_input):
                # Insert the constant node before the first user input
                const_placeholder_node = ep.graph.placeholder(constant_tensor_fqn)
                const_placeholder_node.meta = copy.deepcopy(node.meta)
                const_placeholder_node.meta["val"] = fake_mode.from_tensor(
                    constant_tensor
                )
                node.replace_all_uses_with(const_placeholder_node)
                ep.graph.erase_node(node)

                # Add the constant as a buffer to the graph signature
                inputs_to_buffers[const_placeholder_node.name] = constant_tensor_fqn
                buffers.append(constant_tensor_fqn)
                ep.state_dict[constant_tensor_fqn] = constant_tensor

    ep.graph_module.recompile()
    return ep
