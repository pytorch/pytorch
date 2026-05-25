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
    if not isinstance(custom_obj_meta, CustomObjArgument):
        raise AssertionError(
            f"Expected custom_obj_meta to be a CustomObjArgument, but got {type(custom_obj_meta)}"
        )

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

    if not isinstance(func, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
        raise AssertionError(
            f"Expected func to be an OpOverload or HigherOrderOperator, but got {type(func)}"
        )

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
        if user.target is not operator.getitem:
            raise AssertionError(
                f"Expected user target to be operator.getitem, but got {user.target}"
            )
        # getitem(with_effects, 0) is the token node
        if user.args[1] == 0:
            for user_user in list(user.users.keys()):
                if user_user.op == "output":
                    output_tokens.append(user)

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
        if len(schema.returns) != 0:
            raise AssertionError(
                f"Expected schema.returns to be empty, but got {len(schema.returns)} returns"
            )
        if len(new_node.users) != 0:
            raise AssertionError(
                f"Expected new_node to have no users, but got {len(new_node.users)} users"
            )
        new_node.meta["val"] = None


def _replace_invoke_subgraph_node(node, module, output_tokens, input_tokens):
    """Replace an invoke_subgraph node to remove the token argument."""
    if node.args[0].op != "get_attr":
        raise AssertionError(
            f"Expected node.args[0].op to be 'get_attr', but got {node.args[0].op}"
        )
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


def _get_output_node(module):
    output_node = next(reversed(module.graph.find_nodes(op="output")))
    if output_node is None:
        raise AssertionError("Output node not found in graph")
    return output_node


def _get_output_args(module):
    output_node = _get_output_node(module)
    outs = output_node.args[0]
    if not isinstance(outs, tuple):
        raise AssertionError(f"Expected output tuple, got {type(outs)}")
    return outs


def _getitem_source_and_index(node):
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is operator.getitem
        and len(node.args) > 1
        and isinstance(node.args[0], torch.fx.Node)
        and isinstance(node.args[1], int)
    ):
        return node.args[0], node.args[1]
    return None


def _producer_num_token_outputs(module, producer, cond_token_counts):
    if producer.op != "call_function":
        return 0
    if producer.target is with_effects:
        return 1
    if producer.target is torch.ops.higher_order.invoke_subgraph:
        subgraph_node = producer.args[0]
        if subgraph_node.op == "get_attr" and module.get_submodule(
            subgraph_node.target
        ).meta.get("has_with_effects", False):
            return 1
        return 0
    if producer.target is torch.ops.higher_order.cond:
        return _get_cond_token_count(module, producer, cond_token_counts)
    return 0


def _is_definite_token_output(module, node, cond_token_counts):
    getitem = _getitem_source_and_index(node)
    if getitem is None:
        return False
    producer, index = getitem
    return index < _producer_num_token_outputs(module, producer, cond_token_counts)


def _get_cond_token_count(module, node, cond_token_counts):
    cached = cond_token_counts.get(node)
    if cached is not None:
        return cached

    true_graph_node = node.args[1]
    false_graph_node = node.args[2]
    if true_graph_node.op != "get_attr" or false_graph_node.op != "get_attr":
        raise AssertionError(
            "Expected cond branch nodes to be get_attr nodes, "
            f"got {true_graph_node.op} and {false_graph_node.op}"
        )

    definite_token_indices = set()
    for branch_node in node.args[1:3]:
        submod = module.get_submodule(branch_node.target)
        for index, out in enumerate(_get_output_args(submod)):
            if _is_definite_token_output(submod, out, cond_token_counts):
                definite_token_indices.add(index)

    num_tokens = 0
    while num_tokens in definite_token_indices:
        num_tokens += 1
    cond_token_counts[node] = num_tokens
    return num_tokens


def _get_passthrough_cond_tokens(module, num_tokens):
    return {
        out
        for out in _get_output_args(module)[:num_tokens]
        if isinstance(out, torch.fx.Node) and out.op == "placeholder"
    }


def _replace_cond_node(node, module, num_tokens, output_tokens, input_tokens):
    """Remove effect-token inputs and outputs from a cond node."""
    operands = node.args[3]
    if not isinstance(operands, (list, tuple)):
        raise AssertionError(f"Expected cond operands to be a sequence, got {operands}")

    if num_tokens == 0:
        return

    input_tokens.extend(
        token
        for token in operands[:num_tokens]
        if isinstance(token, torch.fx.Node) and token.op == "placeholder"
    )
    node.args = (*node.args[:3], type(operands)(operands[num_tokens:]))
    if "val" in node.meta and isinstance(node.meta["val"], (list, tuple)):
        node.meta["val"] = node.meta["val"][num_tokens:]

    for user in list(node.users.keys()):
        if user.target is not operator.getitem:
            raise AssertionError(
                f"Expected user target to be operator.getitem, but got {user.target}"
            )
        if user.args[1] >= num_tokens:
            user.args = (node, user.args[1] - num_tokens)
        else:
            for user_user in list(user.users.keys()):
                if user_user.op == "output":
                    output_tokens.append(user)


def _collect_passthrough_cond_tokens(passthrough_tokens, output_tokens, input_tokens):
    if not passthrough_tokens:
        return

    input_tokens.extend(passthrough_tokens)
    output_tokens.extend(passthrough_tokens)


def _remove_effect_tokens(ep: ExportedProgram) -> ExportedProgram:
    """
    Removes the existence of tokens from the exported program, including:
    - Removes the input and output tokens
    - Replaces with_effects(token, func, args) with just func(args)

    This function does an inplace modification on the given ExportedProgram.
    """
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

    def module_has_with_effects(module):
        if module.meta.get("has_with_effects", False):
            return True

        for node in module.graph.nodes:
            if node.target is torch.ops.higher_order.cond:
                for branch_node in node.args[1:3]:
                    if branch_node.op != "get_attr":
                        continue
                    if module_has_with_effects(getattr(module, branch_node.target)):
                        module.meta["has_with_effects"] = True
                        return True
            elif node.target is torch.ops.higher_order.invoke_subgraph:
                subgraph_node = node.args[0]
                if subgraph_node.op == "get_attr" and module_has_with_effects(
                    getattr(module, subgraph_node.target)
                ):
                    module.meta["has_with_effects"] = True
                    return True

        return False

    module_has_with_effects(ep.graph_module)

    cond_token_counts = {}
    cond_branch_token_counts = {}
    for prefix, module in ep.graph_module.named_modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if node.target is not torch.ops.higher_order.cond:
                continue
            num_tokens = _get_cond_token_count(module, node, cond_token_counts)
            if num_tokens == 0:
                continue
            for branch_node in node.args[1:3]:
                if branch_node.op == "get_attr":
                    qualified_branch_name = (
                        f"{prefix}.{branch_node.target}"
                        if prefix
                        else branch_node.target
                    )
                    cond_branch_token_counts[qualified_branch_name] = max(
                        num_tokens,
                        cond_branch_token_counts.get(qualified_branch_name, 0),
                    )

    passthrough_cond_tokens = {}
    for name, module in ep.graph_module.named_modules():
        if (
            isinstance(module, torch.fx.GraphModule)
            and name in cond_branch_token_counts
        ):
            passthrough_cond_tokens[name] = _get_passthrough_cond_tokens(
                module, cond_branch_token_counts[name]
            )

    # Process each module with the replace hook to ensure graph signature is updated
    with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
        for name, module in ep.graph_module.named_modules():
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
                elif node.target is torch.ops.higher_order.cond:
                    _replace_cond_node(
                        node,
                        module,
                        cond_token_counts.get(node, 0),
                        output_tokens,
                        input_tokens,
                    )

            if name in passthrough_cond_tokens:
                _collect_passthrough_cond_tokens(
                    passthrough_cond_tokens[name], output_tokens, input_tokens
                )

            # Remove tokens from the output node
            if len(output_tokens) > 0:
                output_node = _get_output_node(module)
                output_args = output_node.args[0]
                if len(output_args) < len(output_tokens):
                    raise AssertionError(
                        f"{output_args} output arguments found\n"
                        f"{output_tokens} output tokens found\n"
                        f"{module.graph}"
                    )
                output_tokens_set = set(output_tokens)
                output_node.args = (
                    tuple(out for out in output_args if out not in output_tokens_set),
                )

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
            if not isinstance(inp.arg, TokenArgument):
                raise AssertionError(
                    f"Expected inp.arg to be a TokenArgument, but got {type(inp.arg)}"
                )
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

    if num_tokens != num_out_tokens:
        raise AssertionError(
            f"Number of input tokens ({num_tokens}) does not match output tokens ({num_out_tokens})"
        )

    return ep
