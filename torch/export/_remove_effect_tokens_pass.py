# mypy: allow-untyped-defs
import operator
from typing import List

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


def _remove_effect_tokens_from_graph_helper(
    ep, num_tokens, input_token_names, output_token_names
):
    inputs_to_lifted_custom_objs = ep.graph_signature.inputs_to_lifted_custom_objs

    output_node = None
    with_effect_nodes: List[torch.fx.Node] = []

    # Output node need to check its args agianst output_token_names (collected from output_spec)
    # Therefore, we only need to find the top-levele output node
    output_node = next(reversed(ep.graph_module.graph.find_nodes(op="output")))
    for module in ep.graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue

        for node in module.graph.nodes:
            if not (node.op == "call_function" and node.target is with_effects):
                continue

            with_effect_nodes.append(node)

    # Remove tokens from outputs
    assert output_node is not None
    output_args = output_node.args[0]
    assert len(output_args) >= num_tokens
    out_token_nodes = output_args[:num_tokens]
    output_node.args = (tuple(output_args[num_tokens:]),)
    for out_token in out_token_nodes:
        assert out_token.name in output_token_names
        out_token.users.clear()
        ep.graph.erase_node(out_token)

    # Replace with_effects(token, func, args) with just func(args)
    for node in reversed(with_effect_nodes):
        func = node.args[1]
        assert isinstance(func, (torch._ops.OpOverload, torch._ops.HigherOrderOperator))

        if func == torch.ops.higher_order.call_torchbind:
            custom_obj_meta = node.args[2].meta["val"]
            assert isinstance(custom_obj_meta, CustomObjArgument)
            if custom_obj_meta.fake_val:
                custom_obj = custom_obj_meta.fake_val
            elif node.args[2].name in inputs_to_lifted_custom_objs:
                custom_obj = ep.constants[
                    inputs_to_lifted_custom_objs[node.args[2].name]
                ]
            else:
                raise RuntimeError(f"Unable to find custom obj for node {node}")
            schema = _get_schema(func, (custom_obj,) + node.args[3:])
        else:
            schema = _get_schema(func, node.args[2:])

        with ep.graph.inserting_before(node):
            new_node = ep.graph.call_function(func, node.args[2:], node.kwargs)
        for k, v in node.meta.items():
            new_node.meta[k] = v

        node.replace_all_uses_with(new_node)

        # Update user getitem nodes
        for user in list(new_node.users.keys()):
            assert user.target == operator.getitem
            # getitem(with_effects, 0) == token
            if user.args[1] == 0:
                ep.graph.erase_node(user)

        if len(schema.returns) == 1:
            # If the function has 1 return then it will just directly return the
            # result -- we don't need a getitem. So we can replace all the
            # getitem(with_effects, 1) with just the note itself.
            for user in list(new_node.users.keys()):
                assert user.args[1] == 1
                user.replace_all_uses_with(new_node)

            new_node.meta["val"] = node.meta["val"][1]
        elif len(schema.returns) > 1:
            # If the function has more than 1 return then since we got rid of
            # the 1st return value (the token), we need to bump all the other
            # getitem calls by 1 down
            for user in list(new_node.users.keys()):
                assert user.args[1] >= 1
                user.args = (user.args[0], user.args[1] - 1)

            new_node.meta["val"] = node.meta["val"][1:]
        else:
            assert len(schema.returns) == 0
            assert len(new_node.users) == 0
            new_node.meta["val"] = None

        ep.graph.erase_node(node)

    # Remove tokens from inputs
    placeholders = [node for node in ep.graph.nodes if node.op == "placeholder"]
    assert len(placeholders) >= num_tokens
    inp_token_nodes = placeholders[:num_tokens]
    for inp_token in inp_token_nodes:
        assert inp_token.name in input_token_names
        ep.graph.erase_node(inp_token)

    ep.graph.eliminate_dead_code()


def _remove_effect_tokens(ep: ExportedProgram) -> ExportedProgram:
    """
    Removes the existance of tokens from the exported program, including:
    - Removes the input and output tokens
    - Replaces with_effects(token, func, args) with just func(args)

    This function does an inplace modification on the given ExportedProgram.
    """
    num_tokens: int = 0
    input_token_names: List[str] = []
    new_input_specs: List[InputSpec] = []
    for inp in ep.graph_signature.input_specs:
        if inp.kind == InputKind.TOKEN:
            num_tokens += 1
            assert isinstance(inp.arg, TokenArgument)
            input_token_names.append(inp.arg.name)
        else:
            new_input_specs.append(inp)

    num_out_tokens: int = 0
    new_output_specs: List[OutputSpec] = []
    output_token_names: List[OutputSpec] = []
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

    with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
        _remove_effect_tokens_from_graph_helper(
            ep, num_tokens, input_token_names, output_token_names
        )

    return ep
