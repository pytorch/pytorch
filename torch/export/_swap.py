import logging
import operator
import types
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.export.exported_program import (
    ConstantArgument,
    ExportedProgram,
    ModuleCallSignature,
)
from torch.fx.passes.tools_common import legalize_graph, NodeList
from torch.fx.passes.utils.fuser_utils import erase_nodes, fuse_as_graphmodule


log = logging.getLogger(__name__)


def _get_getitem_users(node: torch.fx.Node) -> Set[torch.fx.Node]:
    node_users = list(node.users.keys())
    getitem_users = set()
    for user in node_users:
        if user.op == "output":
            continue

        assert (
            user.op == "call_function" and user.target == operator.getitem
        ), f"Expected getitem node as user for {node}, instead got {user}"
        getitem_users.update(list(user.users.keys()))
    return getitem_users


def _try_remove_connecting_pytrees(curr_module_node: torch.fx.Node) -> None:
    """
    We want to try to remove extraneous pytree flatten/unflatten calls between modules
    calls. Instead of having the following:
    graph():
        ...
        %foo : [num_users=1] = call_module[target=foo](args = (%getitem_1, %getitem_2), kwargs = {})
        %tree_flatten_spec : [num_users=1] = call_function[target=torch.fx._pytree.tree_flatten_spec](args = (%foo, %_spec_1), kwargs = {})
        %getitem_4 : [num_users=1] = call_function[target=operator.getitem](args = (%tree_flatten_spec, 0), kwargs = {})
        %tree_unflatten_1 : [num_users=2] = call_function[target=torch.utils._pytree.tree_unflatten](args = ([%getitem_4], %_spec_2), kwargs = {})
        %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%tree_unflatten_1, 0), kwargs = {})
        %getitem_7 : [num_users=0] = call_function[target=operator.getitem](args = (%tree_unflatten_1, 1), kwargs = {})
        %getitem_6 : [num_users=1] = call_function[target=operator.getitem](args = (%getitem_5, 0), kwargs = {})
        %bar : [num_users=1] = call_module[target=bar](args = (%getitem_6,), kwargs = {})
        ...

    We could do the following, if we know that all the outputs of `foo` feed into `bar`:
    graph():
        ...
        %foo : [num_users=1] = call_module[target=foo](args = (%getitem_1, %getitem_2), kwargs = {})
        %bar : [num_users=1] = call_module[target=bar](args = (%getitem_6,), kwargs = {})
        ...

    Currently this optimization only works for the case where all of the outputs
    of `foo` go directly into `bar`, and `bar` has no other inputs.
    """  # noqa: B950

    log.debug("Trying to remove pytrees for module call %s", curr_module_node)

    curr_module_users = list(curr_module_node.users.keys())
    assert (
        len(curr_module_users) == 1
    ), f"Expected only one user for module node, instead got {list(curr_module_users)}"
    flatten_node = curr_module_users[0]
    assert (
        flatten_node.op == "call_function"
        and flatten_node.target == fx_pytree.tree_flatten_spec
    )

    flatten_getitem_users = _get_getitem_users(flatten_node)
    if len(flatten_getitem_users) != 1:
        log.debug(
            "More than one user found for flatten node, %s: %s. "
            "Unable to fuse it with another unflatten call.",
            flatten_node,
            flatten_getitem_users,
        )
        return

    unflatten_node = next(iter(flatten_getitem_users))
    if not (
        unflatten_node.op == "call_function"
        and unflatten_node.target == pytree.tree_unflatten
    ):
        log.debug(
            "Flatten node %s's user is not a pytree.tree_unflatten. "
            "Instead it is: %s. Passing...",
            flatten_node,
            unflatten_node,
        )
        return

    for i, arg in enumerate(unflatten_node.args[0]):  # type: ignore[union-attr,arg-type]
        if arg not in flatten_node.users:
            log.debug(
                "Module %s's outputs are not all directly used as inputs to "
                "the subsequent module. Unable to fuse the connecting "
                "flatten/unflatten. The inputs to the subsequent module are: %s. ",
                curr_module_node,
                unflatten_node.args[0],
            )
            return

        if not (
            arg.op == "call_function"
            and arg.target == operator.getitem
            and arg.args[1] == i
        ):
            log.debug(
                "Module %s's outputs are not all directly used in the same "
                "order as outputted. Unable to fuse the connecting "
                "flatten/unflatten. The inputs to the "
                "subsequent module are: %s. ",
                curr_module_node,
                unflatten_node.args[0],
            )
            return

    # Unflatten has two levels of getitem, because it gets the args and kwargs
    unflatten_getitem_getitem_users = set()
    unflatten_getitem_users = _get_getitem_users(unflatten_node)
    for unflatten_getitem_user in unflatten_getitem_users:
        unflatten_getitem_getitem_users.update(
            list(unflatten_getitem_user.users.keys())
        )

    if len(unflatten_getitem_getitem_users) != 1:
        log.debug(
            "More than one user found for unflatten node, %s: %s. "
            "Unable to fuse it with another flatten call.",
            unflatten_node,
            unflatten_getitem_getitem_users,
        )
        return

    next_module_node = next(iter(unflatten_getitem_getitem_users))
    if not (next_module_node.op == "call_module"):
        log.debug(
            "Unflatten node %s's user is not a call_module. "
            "Instead it is: %s. Passing...",
            unflatten_node,
            next_module_node,
        )
        return

    # Directly put the outputs of the current module into the next module
    next_module_node.args = (curr_module_node,)


def _remove_extraneous_pytrees(gm: torch.fx.GraphModule) -> None:
    """
    Remove extraneous pytree flatten/unflatten calls.

    We try a couple of optimizations here:
        1. Remove pytree flatten/unflatten calls between modules
        2. TODO: Remove module's in_spec + initial unflatten call
        3. TODO: Remove module's out_spec + final flatten call
    """

    for node in gm.graph.nodes:
        if node.op == "call_module":
            _try_remove_connecting_pytrees(node)

    gm.graph.eliminate_dead_code()


def _construct_inputs(
    gm: torch.fx.GraphModule,
    signature: ModuleCallSignature,
    node_name_map: Dict[str, torch.fx.Node],
) -> Tuple[List[torch.fx.Node], Dict[str, torch.fx.Node]]:
    tree_unflatten_args: List[Optional[torch.fx.Node]] = []
    for input_ in signature.inputs:
        if isinstance(input_, ConstantArgument) and input_.value is None:
            # Constants should be directly embedded into the graph and not used
            # as inputs
            tree_unflatten_args.append(None)
        elif input_.name not in node_name_map:
            # For unused inputs
            tree_unflatten_args.append(None)
        else:
            tree_unflatten_args.append(node_name_map[input_.name])

    # Insert unflatten call
    from .unflatten import _generate_unflatten

    unflatten_node = _generate_unflatten(gm, tree_unflatten_args, signature.in_spec)

    assert signature.in_spec.num_children == 2

    args_spec = signature.in_spec.children_specs[0]
    assert args_spec.context is None
    args_node = gm.graph.call_function(operator.getitem, (unflatten_node, 0))
    args_nodes = [
        gm.graph.call_function(operator.getitem, (args_node, i))
        for i in range(args_spec.num_children)
    ]

    kwargs_spec = signature.in_spec.children_specs[1]
    assert kwargs_spec.context is not None
    kwargs_node = gm.graph.call_function(operator.getitem, (unflatten_node, 1))
    kwargs_nodes = {
        k: gm.graph.call_function(operator.getitem, (kwargs_node, k))
        for k in kwargs_spec.context
    }
    return args_nodes, kwargs_nodes


def _insert_call_module(
    gm: torch.fx.GraphModule,
    args_nodes: List[torch.fx.Node],
    kwargs_nodes: Dict[str, torch.fx.Node],
    module_to_swap: torch.nn.Module,
    name: str,
) -> torch.fx.Node:
    from .unflatten import _assign_attr, _AttrKind

    _assign_attr(module_to_swap, gm, name, _AttrKind.MODULE)
    module_node = gm.graph.call_module(name, tuple(args_nodes), kwargs_nodes)  # type: ignore[arg-type]
    return module_node


def _deconstruct_outputs(
    gm: torch.fx.GraphModule,
    signature: ModuleCallSignature,
    module_node: torch.fx.Node,
    node_name_map: Dict[str, torch.fx.Node],
    orig_outputs: Tuple[torch.fx.Node, ...],
) -> None:
    from .unflatten import _generate_flatten_spec

    flatten_node = _generate_flatten_spec(gm, module_node, signature.out_spec)

    for i, orig_output in enumerate(orig_outputs):
        # Use Proxy to record getitem access.
        proxy_out = torch.fx.Proxy(flatten_node)[i].node  # type: ignore[index]
        orig_output.replace_all_uses_with(proxy_out, propagate_meta=True)

        node_name_map[orig_output.name] = proxy_out


def _swap_module_helper(
    gm: torch.fx.GraphModule,
    modules_to_swap: Dict[str, torch.nn.Module],
    module_call_graph: Dict[str, ModuleCallSignature],
) -> torch.fx.GraphModule:
    log.debug("Starting graph:")
    log.debug(gm.graph)

    legalize_graph(gm)

    partitions: Dict[str, NodeList] = defaultdict(list)

    node_name_map: Dict[str, torch.fx.Node] = {
        node.name: node for node in gm.graph.nodes
    }

    # TODO: Handle the duplicate module case
    for node in gm.graph.nodes:
        if nn_module_stack := node.meta.get("nn_module_stack"):
            for path, _ in nn_module_stack.values():
                if path in modules_to_swap:
                    partitions[path].append(node)
                    break

    for name, nodes in partitions.items():
        """
        Given a graph like the following, and we want to swap out the submodule "foo":
        graph():
            %x : [num_users=1] = placeholder[target=x]
            %y : [num_users=2] = placeholder[target=y]
            %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%y, %x), kwargs = {}), nn_module_stack = {"foo": ("foo", torch.nn.Module)}
            %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%y, %add), kwargs = {}), nn_module_stack = {"bar": ("bar", torch.nn.Module)}
            return (sub,)

        We will first partition out foo's subgraph:
        graph():
            %x : [num_users=1] = placeholder[target=x]
            %y : [num_users=2] = placeholder[target=y]
            %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%y, %x), kwargs = {})
            return add

        And then insert an unflatten + call_module + flatten to replace the subgraph:
        graph():
            %x : [num_users=1] = placeholder[target=x]
            %y : [num_users=1] = placeholder[target=y]

            %_spec_0 : [num_users=1] = get_attr[target=_spec_0]
            %tree_unflatten : [num_users=2] = call_function[target=torch.utils._pytree.tree_unflatten](args = ([%x, %y], %_spec_0), kwargs = {})
            %getitem : [num_users=2] = call_function[target=operator.getitem](args = (%tree_unflatten, 0), kwargs = {})
            %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%getitem, 0), kwargs = {})
            %getitem_2 : [num_users=1] = call_function[target=operator.getitem](args = (%getitem, 1), kwargs = {})
            %getitem_3 : [num_users=0] = call_function[target=operator.getitem](args = (%tree_unflatten, 1), kwargs = {})
            %foo : [num_users=0] = call_module[target=foo](args = (%getitem_1, %getitem_2), kwargs = {})
            %_spec_1 : [num_users=1] = get_attr[target=_spec_1]
            %tree_flatten_spec : [num_users=1] = call_function[target=torch.fx._pytree.tree_flatten_spec](args = (None, %_spec_1), kwargs = {})
            %getitem_4 : [num_users=1] = call_function[target=operator.getitem](args = (%tree_flatten_spec, 0), kwargs = {})

            %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%y, %getitem_4), kwargs = {})
            return (%sub,)

        The `tree_unflatten` call will construct tensor inputs into the input
        format needed by the swapped eager module.
        The `call_module` node should now reference the swapped torch.nn.Module.
        The `tree_flatten_spec` call will deconstruct the eager outputs of the
        swapped module into tensors.
        """  # noqa: B950

        submod_name = name.replace(".", "_")
        sub_gm, orig_inputs, orig_outputs = fuse_as_graphmodule(
            gm, nodes, f"fused_{submod_name}"
        )

        log.debug("Fused subgraph nodes:")
        log.debug(sub_gm.graph)

        signature: ModuleCallSignature = module_call_graph[name]

        args_nodes, kwargs_nodes = _construct_inputs(gm, signature, node_name_map)
        module_node = _insert_call_module(
            gm, args_nodes, kwargs_nodes, modules_to_swap[name], name
        )
        _deconstruct_outputs(gm, signature, module_node, node_name_map, orig_outputs)

        erase_nodes(gm, nodes)

        log.debug("Swapped graph:")
        log.debug(gm.graph)

    legalize_graph(gm)

    log.debug("Before removing extraneous pytrees:")
    log.debug(gm.graph)

    _remove_extraneous_pytrees(gm)
    log.debug("After removing extraneous pytrees:")
    log.debug(gm.graph)

    gm.recompile()

    return gm


def _fix_input_output_signature(
    gm: torch.fx.GraphModule, signature: ModuleCallSignature
) -> None:
    """
    Given the unlifted module from calling ep.module(), we want to remove the
    pytree processing from the graph module's PyTreeCodeGen and instead make it
    nodes inside of the graph. This allows us to do some optimizations, like
    remove these pytree calls if it is unnecessary, and makes the PyTree part
    more obvious to graph passes.
    """
    from torch.export.unflatten import _generate_flatten, _generate_unflatten

    # Remove the registered pytree codegen because we will take care of it
    # through inserting pytree nodes into the graph
    gm.graph._codegen = torch.fx.graph.CodeGen()

    old_placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]

    new_placeholders = []
    forward_arg_names = signature.forward_arg_names
    if forward_arg_names is None:
        forward_arg_names = []
        assert signature.in_spec.num_children == 2
        arg_spec = signature.in_spec.children_specs[0]
        kwarg_spec = signature.in_spec.children_specs[1]
        assert arg_spec.type == tuple
        assert kwarg_spec.type == dict
        for i in range(arg_spec.num_children):
            forward_arg_names.append(f"arg_{i}")
        forward_arg_names.extend(kwarg_spec.context)

    for arg in forward_arg_names:
        with gm.graph.inserting_before(old_placeholders[0]):
            new_placeholders.append(gm.graph.placeholder(arg))

    # Insert flatten call for the inputs
    with gm.graph.inserting_before(old_placeholders[0]):
        flat_node = _generate_flatten(gm, tuple(new_placeholders))
        for i, old_placeholder in enumerate(old_placeholders):
            old_placeholder.op = "call_function"
            old_placeholder.target = operator.getitem
            old_placeholder.args = (flat_node, i)

    # Insert unflatten call for the outputs
    output_node = next(node for node in gm.graph.nodes if node.op == "output")
    with gm.graph.inserting_before(output_node):
        unflat = _generate_unflatten(gm, output_node.args[0], signature.out_spec)
        output_node.args = (unflat,)

    gm.recompile()


def _swap_modules(
    ep: ExportedProgram, modules_to_swap: Dict[str, torch.nn.Module]
) -> torch.fx.GraphModule:
    """
    Unlifts the given ExportedProgram into a fx.GraphModule, and then swaps
    previously traced modules with new eager modules specified. Returns a
    fx.GraphModule with a custom forward function.

    Args:
        ep (ExportedProgram): Exported program to modify
        modules_to_swap (Dict[str, torch.nn.Module]): Mapping from module fqn to
            eager module to swap with. The specified module fqn should have also
            been specified in the `preserve_module_call_signature` argument to
            torch.export so that we know how to restore the calling convention
            to this argument.
        run_with_interpreter: Whether or not to run the graph using
            fx.Interpreter. Setting to true will help result in better error
            messages and easier debugging, but it has found to result in a QPS
            drop.
    """
    module_call_graph = {
        entry.fqn: entry.signature for entry in ep.module_call_graph if entry.signature
    }

    gm = ep.module()
    gm.validate_inputs = False  # type: ignore[assignment]
    gm.graph.eliminate_dead_code()
    assert isinstance(gm, torch.fx.GraphModule)
    _fix_input_output_signature(gm, ep.module_call_graph[0].signature)

    gm.module_call_graph = ep.module_call_graph
    gm.train = types.MethodType(type(gm).train, gm)  # type: ignore[assignment]
    gm.eval = types.MethodType(type(gm).eval, gm)  # type: ignore[assignment]

    assert isinstance(gm, torch.fx.GraphModule)
    gm = _swap_module_helper(gm, modules_to_swap, module_call_graph)

    return gm
