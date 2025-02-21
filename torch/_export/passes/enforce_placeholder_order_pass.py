# mypy: allow-untyped-defs

from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import InputKind


def enforce_placeholder_order_pass(
    ep: ExportedProgram,
):
    # Enforce the order of placeholder nodes in the graph module to be:
    # token -> parameter -> buffer (persistent) -> buffer (non-persistent)
    # -> tensor_constant -> custom_obj -> user_inputs

    gm = ep.graph_module
    graph_signature = gm.graph_signature

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

    with gm.graph.inserting_before():
        for input_node in reversed(reordered_input_nodes):
            new_node = gm.graph.node_copy(input_node)
            new_node.name = input_node.name
            input_node.replace_all_uses_with(new_node)
            gm.graph.erase_node(input_node)

    gm.recompile()
    graph_signature.input_specs = reordered_input_specs

    ep.graph_module = gm
    ep.graph_signature = graph_signature

    return ep
