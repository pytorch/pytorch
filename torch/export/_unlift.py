# mypy: allow-untyped-defs
import copy
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch._export.utils import _check_input_constraints_for_graph
from torch.export.unflatten import _assign_attr, _AttrKind
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from ._remove_effect_tokens_pass import _remove_effect_tokens

from .exported_program import (
    ExportedProgram,
    ExportGraphSignature,
    InputKind,
    OutputKind,
)


@torch._dynamo.disable
def _check_input_constraints_pre_hook(self, *args, **kwargs):
    flat_args_with_path, received_spec = pytree.tree_flatten_with_path(args)

    if received_spec != self._in_spec:
        raise ValueError(  # noqa: B904
            "Trying to flatten user inputs with exported input tree spec: \n"
            f"{self._in_spec}\n"
            "but actually got inputs with tree spec of: \n"
            f"{received_spec}"
        )

    return _check_input_constraints_for_graph(
        [node for node in self.graph.nodes if node.op == "placeholder"],
        flat_args_with_path,
        self.range_constraints,
    )


def _unlift_inputs_as_getattr(
    gm: torch.fx.GraphModule,
    lifted_inputs: List[Optional[str]],
) -> Tuple[Dict[str, torch.fx.Node], Dict[str, torch.fx.Node]]:
    """
    Unlift inputs referring to params/buffers/constants as getattr nodes in the
    graph
    """
    unlifted_name_to_node = {}
    input_name_to_node = {}

    placeholder_nodes = [node for node in gm.graph.nodes if node.op == "placeholder"]
    assert len(lifted_inputs) == len(placeholder_nodes)
    for input_node, lifted_node in zip(placeholder_nodes, lifted_inputs):
        if lifted_node is None:
            input_name_to_node[input_node.name] = input_node

        else:
            with gm.graph.inserting_after(input_node):
                getattr_node = gm.graph.get_attr(lifted_node)
                input_node.replace_all_uses_with(getattr_node)
                metadata = input_node.meta
                gm.graph.erase_node(input_node)
                getattr_node.meta = metadata
                unlifted_name_to_node[lifted_node] = getattr_node

    return unlifted_name_to_node, input_name_to_node


def _insert_copy_for_mutations(
    gm: torch.fx.GraphModule,
    mutated_outputs: List[Optional[str]],
    unlifted_name_to_node: Dict[str, torch.fx.Node],
    input_name_to_node: Dict[str, torch.fx.Node],
) -> None:
    """
    Find the all the buffers and inputs that were mutated and insert copy_
    operators to reflect mutations.
    """
    output_node = None
    for node in gm.graph.nodes:
        if node.op == "output":
            output_node = node
            break
    assert output_node is not None
    outputs = pytree.tree_flatten(output_node.args)[0]
    assert len(outputs) == len(mutated_outputs)

    user_output_nodes = []
    return_nodes_to_copy = {}
    for return_node, mutated_node_name in zip(outputs, mutated_outputs):
        if mutated_node_name is None:
            user_output_nodes.append(return_node)
            continue

        if mutated_node_name in unlifted_name_to_node:
            mutated_node = unlifted_name_to_node[mutated_node_name]
        elif mutated_node_name in input_name_to_node:
            mutated_node = input_name_to_node[mutated_node_name]
        else:
            raise RuntimeError(
                f"Could not find {mutated_node_name} in either buffer or input nodes"
            )

        with gm.graph.inserting_before(output_node):
            copy_node = gm.graph.call_function(
                torch.ops.aten.copy_.default, (mutated_node, return_node)
            )
            return_nodes_to_copy[return_node] = copy_node

    output_args = [
        return_nodes_to_copy[node] if node in return_nodes_to_copy else node
        for node in user_output_nodes
    ]
    with gm.graph.inserting_before(output_node):
        # Only return user outputs
        new_output = gm.graph.output(tuple(output_args))
        output_node.replace_all_uses_with(new_output)
        gm.graph.erase_node(output_node)


def _get_codegen(
    in_spec: pytree.TreeSpec,
    out_spec: Optional[pytree.TreeSpec],
    forward_arg_names: Optional[List[str]] = None,
) -> _PyTreeCodeGen:
    """
    Create the codegen for the graph module based on the in/out specs
    """
    if forward_arg_names:
        names = forward_arg_names
    else:
        if (
            in_spec.type == tuple
            and in_spec.num_children == 2
            and in_spec.children_specs[0].type == tuple
            and in_spec.children_specs[1].type == dict
        ):
            # if in_spec contains the args (tuple) and kwargs (dict)
            names = [f"arg_{i}" for i in range(in_spec.children_specs[0].num_children)]
            # add kwarg names
            names.extend(in_spec.children_specs[1].context)
        else:
            names = [f"arg_{i}" for i in range(in_spec.num_children)]

    return _PyTreeCodeGen(
        _PyTreeInfo(
            names,
            in_spec,
            out_spec,
        )
    )


def _unlift(
    gm: torch.fx.GraphModule,
    lifted_inputs: List[Optional[str]],
    mutated_outputs: List[Optional[str]],
    in_spec: pytree.TreeSpec,
    out_spec: Optional[pytree.TreeSpec],
    state_dict: Dict[str, Any],
    constants: Dict[str, Any],
    forward_arg_names: Optional[List[str]] = None,
):
    """
    Args:
        lifted_inputs: A list matching the graph module's input nodes. For
        an input node that is referring to a lifted parameter/buffer, this
        list will contain the fqn the corresponding attribute. Otherwise, this
        list will contain None. This is used to unlift the lifted parameters as
        get_attr nodes.

        mutated_outputs: A list matching the graph module's output nodes. For
        an output node that is referring to a mutated buffer or user input, this
        list will contain the name of the corresponding buffer or user input
        that needs to be mutated. Otherwise, this list will contain None. This
        is used to re-insert an inplace copy_ operator to copy the mutated
        values back to the original node.
    """
    unlifted_name_to_node, input_name_to_node = _unlift_inputs_as_getattr(
        gm, lifted_inputs
    )
    _insert_copy_for_mutations(
        gm, mutated_outputs, unlifted_name_to_node, input_name_to_node
    )
    gm.graph._codegen = _get_codegen(in_spec, out_spec, forward_arg_names)
    gm.graph.lint()
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def _register_attrs_to_new_gm(
    new_gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    state_dict: Dict[str, Any],
    constants: Dict[str, Any],
) -> None:
    non_persistent_buffers = set(graph_signature.non_persistent_buffers)
    for name in graph_signature.buffers:
        if name in non_persistent_buffers:
            persistent = False
            value = constants[name]
        else:
            persistent = True
            value = state_dict[name]
        _assign_attr(
            value, new_gm, name, attr_kind=_AttrKind.BUFFER, persistent=persistent
        )
    for name in graph_signature.parameters:
        value = state_dict[name]
        _assign_attr(
            value,
            new_gm,
            name,
            attr_kind=_AttrKind.PARAMETER,
        )

    for name in chain(
        graph_signature.lifted_custom_objs, graph_signature.lifted_tensor_constants
    ):
        value = constants[name]
        _assign_attr(
            value,
            new_gm,
            name,
            attr_kind=_AttrKind.CONSTANT,
        )


class _StatefulGraphModuleFactory(type):
    """
    Metaclass that ensures a private constructor for _StatefulGraphModule
    """

    def __call__(cls, *args, **kwargs):
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} has no public constructor. "
        )

    def _create(cls, root, graph, range_constraints=None):
        return super().__call__(
            root,
            graph,
            range_constraints=range_constraints,
        )


class _StatefulGraphModule(torch.fx.GraphModule, metaclass=_StatefulGraphModuleFactory):
    def __init__(self, root, graph, range_constraints=None):
        super().__init__(root, graph)
        # Need to fix up non-persistent buffers.
        self.range_constraints = range_constraints or []


def _create_stateful_graph_module(
    plain_graph_module: torch.fx.GraphModule,
    range_constraints,
    # TODO(suo) this should not be optional, but is since we still ahve
    # capture_pre_autograd_graph grr
    graph_signature: Optional[ExportGraphSignature] = None,
):
    stateful_gm = _StatefulGraphModule._create(
        plain_graph_module,
        plain_graph_module.graph,
        range_constraints=range_constraints,
    )
    stateful_gm.register_forward_pre_hook(
        _check_input_constraints_pre_hook, with_kwargs=True
    )

    if graph_signature is None:
        return stateful_gm
    # Fix up non-persistent buffers. torch.fx does not distinguish between
    # persistent and non-persistent buffers, so we must restore that distinction
    # here.
    for buffer in graph_signature.non_persistent_buffers:
        _assign_attr(
            plain_graph_module.get_buffer(buffer),
            stateful_gm,
            buffer,
            attr_kind=_AttrKind.BUFFER,
            persistent=False,
        )

    return stateful_gm


def _unlift_exported_program_lifted_states(ep: ExportedProgram) -> torch.nn.Module:
    ep = _remove_effect_tokens(ep)
    new_gm = torch.fx.GraphModule(ep.graph_module, copy.deepcopy(ep.graph))
    _register_attrs_to_new_gm(new_gm, ep.graph_signature, ep.state_dict, ep.constants)
    forward_arg_names = ep.graph_module.meta.get("forward_arg_names")

    lifted_inputs: List[Optional[str]] = [
        (
            in_spec.target
            if in_spec.kind
            in (
                InputKind.BUFFER,
                InputKind.CONSTANT_TENSOR,
                InputKind.PARAMETER,
                InputKind.CUSTOM_OBJ,
            )
            else None
        )
        for in_spec in ep.graph_signature.input_specs
    ]

    mutated_outputs: List[Optional[str]] = [
        (
            out_spec.target
            if out_spec.kind
            in (OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION)
            else None
        )
        for out_spec in ep.graph_signature.output_specs
    ]

    new_gm = _unlift(
        new_gm,
        lifted_inputs,
        mutated_outputs,
        ep.call_spec.in_spec,
        ep.call_spec.out_spec,
        ep.state_dict,
        ep.constants,
        forward_arg_names=forward_arg_names,
    )
    unlift_gm = _create_stateful_graph_module(
        new_gm, ep.range_constraints, ep.graph_signature
    )
    unlift_gm.meta.update(ep.graph_module.meta)
    return unlift_gm
