# mypy: allow-untyped-defs
import operator

import torch
from torch._higher_order_ops.effects import _get_schema, with_effects

from .exported_program import ExportedProgram
from .graph_signature import (
    CustomObjArgument,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TokenArgument,
)


def _get_custom_obj_for_node(node, inputs_to_lifted_custom_objs, constants):
    """Extract the custom object from a node's arguments."""
    custom_obj_node = node
    custom_obj_meta = custom_obj_node.meta["val"]  # type: ignore[union-attr]
    assert isinstance(custom_obj_meta, CustomObjArgument)

    if custom_obj_meta.fake_val:
        return custom_obj_meta.fake_val
    elif custom_obj_node.name in inputs_to_lifted_custom_objs:  # type: ignore[union-attr]
        return constants[inputs_to_lifted_custom_objs[custom_obj_node.name]]  # type: ignore[union-attr]
    else:
        raise RuntimeError(f"Unable to find custom obj for node {node}")


def _replace_with_effects_node(
    node, ep, inputs_to_lifted_custom_objs, output_tokens, input_tokens, module
):
    """Replace a with_effects node with the underlying function call."""
    # Get the input nodes
    token_node, func, *node_args = node.args
    if token_node.op == "placeholder":
        input_tokens.append(token_node)

    assert isinstance(func, (torch._ops.OpOverload, torch._ops.HigherOrderOperator))

    # Get the schema for the function
    if func is torch.ops.higher_order.call_torchbind:
        custom_obj = _get_custom_obj_for_node(
            node_args[0], inputs_to_lifted_custom_objs, ep.constants
        )
        schema = _get_schema(func, [custom_obj] + node_args[1:])
    else:
        schema = _get_schema(func, node_args)

    # Create the replacement node
    with module.graph.inserting_before(node):
        new_node = module.graph.call_function(func, tuple(node_args), node.kwargs)

    # Update getitem nodes that extract outputs from with_effects
    for user in list(node.users.keys()):
        assert user.target is operator.getitem
        # getitem(with_effects, 0) is the token node
        if user.args[1] == 0:
            for user_user in list(user.users.keys()):
                if user_user.op == "output":
                    output_tokens.append(user)

    # Fix up the getitem nodes based on return count
    if len(schema.returns) == 1:
        # Single return: replace getitem(with_effects, 1) with the node itself
        for user in list(node.users.keys()):
            if user.args[1] == 1:
                user.replace_all_uses_with(new_node)
        new_node.meta["val"] = node.meta["val"][1]
    elif len(schema.returns) > 1:
        # Multiple returns: shift getitem indices down by 1
        for user in list(node.users.keys()):
            if user.args[1] >= 1:
                user.args = (new_node, user.args[1] - 1)
        new_node.meta["val"] = node.meta["val"][1:]
    else:
        # No returns
        assert len(schema.returns) == 0
        assert len(new_node.users) == 0
        new_node.meta["val"] = None

    # Copy metadata from old node to new node
    for k, v in node.meta.items():
        new_node.meta[k] = v
        if k == "unbacked_bindings":
            # Remove the extra layer for effect token
            old_bindings = new_node.meta[k]
            new_bindings = {
                k: path[1:] if path else path for k, path in old_bindings.items()
            }
            new_node.meta[k] = new_bindings


def _replace_invoke_subgraph_node(node, module, output_tokens, input_tokens):
    """Replace an invoke_subgraph node to remove the token argument."""
    assert node.args[0].op == "get_attr"
    submod = getattr(module, node.args[0].target)
    if not submod.meta.get("has_with_effects", False):
        return

    # Remove token from inputs
    subgraph, identifier, token, *operands = node.args
    node.args = (subgraph, identifier, *operands)
    if token.op == "placeholder":
        input_tokens.append(token)

    # Update getitem nodes to account for removed token output
    for user in list(node.users.keys()):
        if user.args[1] >= 1:
            user.args = (node, user.args[1] - 1)
        elif user.args[1] == 0:
            for user_user in list(user.users.keys()):
                if user_user.op == "output":
                    output_tokens.append(user)


def _remove_effect_tokens(ep: ExportedProgram) -> ExportedProgram:
    """
    Removes the existence of tokens from the exported program, including:
    - Removes the input and output tokens
    - Replaces with_effects(token, func, args) with just func(args)

    This function does an inplace modification on the given ExportedProgram.
    """
    print("before", ep)
    inputs_to_lifted_custom_objs = ep.graph_signature.inputs_to_lifted_custom_objs

    # mark submodules with effects as having effects. This will be used in the following pass to remove effects from subgraphs
    for _, module in ep.graph_module.named_modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue

        with_effect_nodes = [
            node for node in module.graph.nodes if node.target is with_effects
        ]
        if len(with_effect_nodes) > 0:
            module.meta["has_with_effects"] = True

    # Process each module with the replace hook to ensure graph signature is updated
    with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
        for _, module in ep.graph_module.named_modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue

            input_tokens = []
            output_tokens = []

            # Process with_effects and invoke_subgraph nodes
            for node in module.graph.nodes:
                if node.target is with_effects:
                    _replace_with_effects_node(
                        node,
                        ep,
                        inputs_to_lifted_custom_objs,
                        output_tokens,
                        input_tokens,
                        module,
                    )
                elif node.target is torch.ops.higher_order.invoke_subgraph:
                    _replace_invoke_subgraph_node(
                        node, module, output_tokens, input_tokens
                    )

            # Remove tokens from the output node
            if len(output_tokens) > 0:
                output_node = next(reversed(module.graph.find_nodes(op="output")))
                output_args = output_node.args[0]
                assert len(output_args) >= len(output_tokens), (
                    f"{output_args} output arguments found\n"
                    f"{output_tokens} output tokens found\n"
                    f"{module.graph}"
                )
                output_node.args = (tuple(output_args[len(output_tokens) :]),)

            module.graph.eliminate_dead_code()

            # Remove tokens from the input placeholders
            for node in module.graph.nodes:
                if node.op == "placeholder" and node in input_tokens:
                    module.graph.erase_node(node)

            module.recompile()

    num_tokens: int = 0
    input_token_names: list[str] = []
    new_input_specs: list[InputSpec] = []
    for inp in ep.graph_signature.input_specs:
        if inp.kind == InputKind.TOKEN:
            num_tokens += 1
            assert isinstance(inp.arg, TokenArgument)
            input_token_names.append(inp.arg.name)
        else:
            new_input_specs.append(inp)

    num_out_tokens: int = 0
    new_output_specs: list[OutputSpec] = []
    output_token_names: list[OutputSpec] = []
    for out in ep.graph_signature.output_specs:
        if out.kind == OutputKind.TOKEN:
            num_out_tokens += 1
            output_token_names.append(out.arg.name)
        else:
            new_output_specs.append(out)

    # Update graph signature
    ep.graph_signature.input_specs = new_input_specs
    ep.graph_signature.output_specs = new_output_specs

    assert num_tokens == num_out_tokens

    print("after", ep)
    return ep
