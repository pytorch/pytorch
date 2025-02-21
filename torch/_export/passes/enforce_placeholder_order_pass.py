# mypy: allow-untyped-defs

import torch
from torch.export.graph_signature import ExportGraphSignature, InputKind


def enforce_placeholder_order_pass(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature
):
    # Enforce the order of placeholder nodes in the graph module to be:
    # token -> parameter -> buffer (persistent) -> buffer (non-persistent)
    # -> tensor_constant -> custom_obj -> user_inputs
    input_nodes = [node for node in gm.graph.nodes if node.op == "placeholder"]
    if len(input_nodes) <= 1:
        return

    reordered_input_nodes = []
    reordered_input_specs = []

    for input_kind in [
        InputKind.TOKEN,
        InputKind.PARAMETER,
        InputKind.BUFFER,
        InputKind.CONSTANT_TENSOR,
        InputKind.CUSTOM_OBJ,
        InputKind.USER_INPUT,
    ]:
        if input_kind == InputKind.BUFFER:
            # Enforce that persistent buffers always come before non-persistent ones
            for input_spec, input_node in zip(graph_signature.input_specs, input_nodes):
                if input_spec.kind == input_kind and input_spec.persistent:
                    reordered_input_nodes.append(input_node)
                    reordered_input_specs.append(input_spec)
            for input_spec, input_node in zip(graph_signature.input_specs, input_nodes):
                if input_spec.kind == input_kind and not input_spec.persistent:
                    reordered_input_nodes.append(input_node)
                    reordered_input_specs.append(input_spec)
        else:
            for input_spec, input_node in zip(graph_signature.input_specs, input_nodes):
                if input_spec.kind == input_kind:
                    reordered_input_nodes.append(input_node)
                    reordered_input_specs.append(input_spec)

    assert len(reordered_input_specs) == len(graph_signature.input_specs)
    assert len(reordered_input_nodes) == len(input_nodes)

    with gm.graph.inserting_before(input_nodes[0]):
        new_placeholder_nodes = []
        for input_node in reordered_input_nodes:
            new_node = gm.graph.placeholder(input_node.name)
            # overwrite new_node.name to be the same as input_node.name
            # in case there is naming collision and suffix is added
            new_node.name = input_node.name
            new_node.meta = input_node.meta
            new_placeholder_nodes.append(new_node)

    node_mapping = dict(zip(reordered_input_nodes, new_placeholder_nodes))

    for node in gm.graph.nodes:
        if node.op != "placeholder":
            node.args = torch.fx.map_arg(node.args, lambda n: node_mapping.get(n, n))
            node.kwargs = torch.fx.map_arg(
                node.kwargs, lambda n: node_mapping.get(n, n)
            )

    for old_node in input_nodes:
        gm.graph.erase_node(old_node)

    gm.recompile()
    graph_signature.input_specs = reordered_input_specs
