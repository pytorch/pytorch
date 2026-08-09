# mypy: allow-untyped-defs
import operator

import torch
from torch._higher_order_ops._effect_token_utils import EffectTokenAnalyzer
from torch._higher_order_ops.effects import _get_schema, with_effects
from torch.utils._pytree import SequenceKey

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

    num_returns = len(schema.returns)

    def normalized_output_index(user):
        index = user.args[1]
        if not isinstance(index, int):
            return None
        if index < 0:
            val = node.meta.get("val")
            output_arity = (
                len(val) if isinstance(val, (list, tuple)) else num_returns + 1
            )
            index += output_arity
        return index if 0 <= index <= num_returns else None

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
        if normalized_output_index(user) == 0:
            for user_user in list(user.users.keys()):
                if user_user.op == "output":
                    output_tokens.append(user)

    # Copy metadata from old node to new node
    for k, v in node.meta.items():
        new_node.meta[k] = v
        if k == "unbacked_bindings":
            new_bindings = {}
            for symbol, path in v.items():
                if path and isinstance(path[0], SequenceKey):
                    if path[0].idx == 0:
                        continue
                    if num_returns == 1:
                        path = path[1:]
                    else:
                        path = (SequenceKey(path[0].idx - 1), *path[1:])
                new_bindings[symbol] = path
            new_node.meta[k] = new_bindings

    # Fix up the getitem nodes based on return count
    if num_returns == 1:
        # Single return: replace getitem(with_effects, 1) with the node itself
        for user in list(node.users.keys()):
            if normalized_output_index(user) == 1:
                user.replace_all_uses_with(new_node)
        new_node.meta["val"] = node.meta["val"][1]
    elif num_returns > 1:
        # Multiple returns: shift getitem indices down by 1
        for user in list(node.users.keys()):
            index = normalized_output_index(user)
            if index is not None and index >= 1:
                user.args = (new_node, index - 1)
        new_node.meta["val"] = node.meta["val"][1:]
    else:
        # No returns
        if num_returns != 0:
            raise AssertionError(
                f"Expected schema.returns to be empty, but got {num_returns} returns"
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
    submod = module.get_submodule(node.args[0].target)
    if not submod.meta.get("has_with_effects", False):
        return

    # Remove token from inputs
    subgraph, identifier, token, *operands = node.args
    node.args = (subgraph, identifier, *operands)
    if token.op == "placeholder":
        input_tokens.append(token)

    if "val" in node.meta and isinstance(node.meta["val"], (list, tuple)):
        node.meta["val"] = node.meta["val"][1:]
    if "unbacked_bindings" in node.meta:
        shifted_bindings = {}
        for symbol, path in node.meta["unbacked_bindings"].items():
            if path and isinstance(path[0], SequenceKey):
                if path[0].idx == 0:
                    continue
                path = (SequenceKey(path[0].idx - 1), *path[1:])
            shifted_bindings[symbol] = path
        node.meta["unbacked_bindings"] = shifted_bindings

    total_outputs = len(node.meta["val"]) + 1 if "val" in node.meta else None
    # Update getitem nodes to account for removed token output
    for user in list(node.users.keys()):
        index = user.args[1]
        if isinstance(index, int) and index < 0 and total_outputs is not None:
            index += total_outputs
        if isinstance(index, int) and index >= 1:
            user.args = (node, index - 1)
        elif index == 0:
            for user_user in list(user.users.keys()):
                if user_user.op == "output":
                    output_tokens.append(user)


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
    total_outputs = None
    if "val" in node.meta and isinstance(node.meta["val"], (list, tuple)):
        total_outputs = len(node.meta["val"])
        node.meta["val"] = node.meta["val"][num_tokens:]
    if "unbacked_bindings" in node.meta:
        shifted_bindings = {}
        for symbol, path in node.meta["unbacked_bindings"].items():
            if path and isinstance(path[0], SequenceKey):
                if path[0].idx < num_tokens:
                    continue
                path = (SequenceKey(path[0].idx - num_tokens), *path[1:])
            shifted_bindings[symbol] = path
        node.meta["unbacked_bindings"] = shifted_bindings

    for user in list(node.users.keys()):
        if user.target is not operator.getitem:
            raise AssertionError(
                f"Expected user target to be operator.getitem, but got {user.target}"
            )
        index = user.args[1]
        if isinstance(index, int) and index < 0 and total_outputs is not None:
            index += total_outputs
        if isinstance(index, int) and index >= num_tokens:
            user.args = (node, index - num_tokens)
        elif isinstance(index, int) and 0 <= index < num_tokens:
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
    if not ep.graph_signature.input_tokens and not ep.graph_signature.output_tokens:
        return ep

    inputs_to_lifted_custom_objs = ep.graph_signature.inputs_to_lifted_custom_objs

    effectful_module_cache = {}
    visiting_modules = set()

    def hop_submodule(module, node):
        if (
            isinstance(node, torch.fx.Node)
            and node.op == "get_attr"
            and isinstance(node.target, str)
        ):
            submodule = module.get_submodule(node.target)
            if isinstance(submodule, torch.fx.GraphModule):
                return submodule
        return None

    def module_has_with_effects(module):
        module_id = id(module)
        if module_id in effectful_module_cache:
            return effectful_module_cache[module_id]
        if module_id in visiting_modules:
            return False
        visiting_modules.add(module_id)

        has_with_effects = any(
            node.target is with_effects for node in module.graph.nodes
        )

        for node in module.graph.nodes:
            if node.target is torch.ops.higher_order.cond:
                for branch_node in node.args[1:3]:
                    branch = hop_submodule(module, branch_node)
                    if branch is not None and module_has_with_effects(branch):
                        has_with_effects = True
            elif node.target is torch.ops.higher_order.invoke_subgraph:
                subgraph = hop_submodule(module, node.args[0])
                if subgraph is not None and module_has_with_effects(subgraph):
                    has_with_effects = True

        visiting_modules.remove(module_id)
        effectful_module_cache[module_id] = has_with_effects
        if has_with_effects:
            module.meta["has_with_effects"] = True
        return has_with_effects

    module_has_with_effects(ep.graph_module)

    def invoke_subgraph_token_count(module, node):
        subgraph_node = node.args[0]
        if (
            isinstance(subgraph_node, torch.fx.Node)
            and subgraph_node.op == "get_attr"
            and isinstance(subgraph_node.target, str)
            and module.get_submodule(subgraph_node.target).meta.get(
                "has_with_effects", False
            )
        ):
            return 1
        return 0

    token_analyzer = EffectTokenAnalyzer(invoke_subgraph_token_count)
    effectful_hop_nodes = set()
    cond_branch_token_counts = {}
    for _, module in ep.graph_module.named_modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if (
                node.target is torch.ops.higher_order.invoke_subgraph
                and invoke_subgraph_token_count(module, node) > 0
            ):
                effectful_hop_nodes.add(node)

            if node.target is not torch.ops.higher_order.cond:
                continue
            num_tokens = token_analyzer.cond_token_count(module, node)
            if num_tokens == 0:
                continue
            effectful_hop_nodes.add(node)
            for branch_node in node.args[1:3]:
                if branch_node.op == "get_attr":
                    branch = module.get_submodule(branch_node.target)
                    cond_branch_token_counts[branch] = max(
                        num_tokens,
                        cond_branch_token_counts.get(branch, 0),
                    )

    passthrough_cond_tokens = {}
    for module, num_tokens in cond_branch_token_counts.items():
        if isinstance(module, torch.fx.GraphModule):
            passthrough_cond_tokens[module] = token_analyzer.passthrough_cond_tokens(
                module, num_tokens
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
                        token_analyzer.cond_token_counts.get(node, 0),
                        output_tokens,
                        input_tokens,
                    )

            if module in passthrough_cond_tokens:
                _collect_passthrough_cond_tokens(
                    passthrough_cond_tokens[module], output_tokens, input_tokens
                )

            # Remove tokens from the output node
            if len(output_tokens) > 0:
                output_node = token_analyzer.output_node(module)
                output_args = token_analyzer.output_args(module)
                if len(output_args) < len(output_tokens):
                    raise AssertionError(
                        f"{output_args} output arguments found\n"
                        f"{output_tokens} output tokens found\n"
                        f"{module.graph}"
                    )
                output_tokens_set = set(output_tokens)
                output_node.args = (
                    type(output_args)(
                        out for out in output_args if out not in output_tokens_set
                    ),
                )

            for node in reversed(module.graph.nodes):
                if (
                    not node.users
                    and node not in effectful_hop_nodes
                    and not node.is_impure()
                ):
                    module.graph.erase_node(node)

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
