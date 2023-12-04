from typing import Dict

import torch
from torch._guards import detect_fake_mode
from torch.export.exported_program import InputKind, InputSpec, TensorArgument


def lift_constant_tensor_pass(gm, graph_signature) -> Dict[str, torch.Tensor]:
    """
    Takes an ExportedProgram and returns the ExportedProgram modified in-place,
    with the constant tensors as buffers.
    """
    if len([node for node in gm.graph.nodes if node.op == "placeholder"]) == 0:
        return {}

    inputs = graph_signature.input_specs
    num_tensor_constants = sum(
        input_specs.kind == InputKind.CONSTANT_TENSOR for input_specs in inputs
    )

    fake_mode = detect_fake_mode(
        tuple(node.meta["val"] for node in gm.graph.nodes if node.op == "placeholder")
    )
    assert fake_mode is not None

    first_user_input_loc, first_user_input = None, None
    for i, node in enumerate(gm.graph.nodes):
        if node.op == "placeholder" and node.name in graph_signature.user_inputs:
            first_user_input = node
            first_user_input_loc = i
            break

    assert first_user_input is not None and first_user_input_loc is not None
    tensor_constants = {}

    for node in gm.graph.nodes:
        if node.op == "get_attr":
            constant_tensor = getattr(gm, node.target)
            if not isinstance(constant_tensor, torch.Tensor):
                continue

            constant_tensor_fqn = f"_lifted_tensor_constant{num_tensor_constants}"
            num_tensor_constants += 1

            with gm.graph.inserting_before(first_user_input):
                # Insert the constant node before the first user input
                const_placeholder_node = gm.graph.placeholder(constant_tensor_fqn)
                for k, v in node.meta.items():
                    const_placeholder_node.meta[k] = v
                const_placeholder_node.meta["val"] = fake_mode.from_tensor(
                    constant_tensor, static_shapes=True
                )
                const_placeholder_node.meta["val"].constant = constant_tensor
                node.replace_all_uses_with(const_placeholder_node)
                gm.graph.erase_node(node)

                # Add the constant as a buffer to the graph signature
                graph_signature.input_specs.insert(
                    first_user_input_loc,
                    InputSpec(
                        kind=InputKind.CONSTANT_TENSOR,
                        arg=TensorArgument(name=const_placeholder_node.name),
                        target=constant_tensor_fqn,
                    ),
                )
                tensor_constants[constant_tensor_fqn] = constant_tensor
                first_user_input_loc += 1

    gm.recompile()
    return tensor_constants
