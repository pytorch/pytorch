import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch._export.utils import _check_input_constraints_pre_hook
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo

from .exported_program import (
    ExportedProgram,
    ExportGraphSignature,
    InputKind,
    OutputKind,
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
                getattr_node = gm.graph.get_attr(lifted_node.replace(".", "_"))
                input_node.replace_all_uses_with(getattr_node)
                metadata = input_node.meta
                gm.graph.erase_node(input_node)
                getattr_node.meta = metadata
                unlifted_name_to_node[lifted_node.replace(".", "_")] = getattr_node

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
    for return_node, mutated_node_name in zip(outputs, mutated_outputs):
        if mutated_node_name is None:
            user_output_nodes.append(return_node)
            continue

        mutated_node_name = mutated_node_name.replace(".", "_")
        if mutated_node_name in unlifted_name_to_node:
            mutated_node = unlifted_name_to_node[mutated_node_name]
        elif mutated_node_name in input_name_to_node:
            mutated_node = input_name_to_node[mutated_node_name]
        else:
            raise RuntimeError(
                f"Could not find {mutated_node_name} in either buffer or input nodes"
            )

        with gm.graph.inserting_before(output_node):
            _ = gm.graph.call_function(
                torch.ops.aten.copy_.default, (mutated_node, return_node)
            )

    with gm.graph.inserting_before(output_node):
        # Only return user outputs
        new_output = gm.graph.output(tuple(user_output_nodes))
        output_node.replace_all_uses_with(new_output)
        gm.graph.erase_node(output_node)


def _get_codegen(
    in_spec: pytree.TreeSpec,
    out_spec: Optional[pytree.TreeSpec],
) -> _PyTreeCodeGen:
    """
    Create the codegen for the graph module based on the in/out specs
    """
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


def _unlift_submod_inputs(
    submod: torch.fx.GraphModule,
    inputs: List[torch.fx.Node],
    toplevel_gm: torch.fx.GraphModule,
    toplevel_unlifted_nodes: List[torch.fx.Node],
) -> Tuple[List[Optional[str]], List[torch.fx.Node]]:
    """
    Given a list of nodes being passed to the the submodule, if any of them
    belong to an input that should be unlifted (ex. parameter/buffer), we should
    remove it from the argument list and register the actual parameter/bffer
    value to the submodule. A later recursive call to _unlift() on the submodule
    parent module will fix the graph inside of the submodule to use unlifted
    inputs.
    """
    submod_lifted_inputs: List[Optional[str]] = []
    real_inputs = []
    for inp in inputs:
        if inp in toplevel_unlifted_nodes:
            assert isinstance(inp.target, str)
            submod_lifted_inputs.append(inp.target)
            if inp.target not in toplevel_gm.state_dict():
                raise RuntimeError("Unable to find value for ", inp.target)

            submod.register_buffer(inp.target, toplevel_gm.state_dict()[inp.target])
        else:
            submod_lifted_inputs.append(None)
            real_inputs.append(inp)

    return submod_lifted_inputs, real_inputs


def _unlift(
    gm: torch.fx.GraphModule,
    lifted_inputs: List[Optional[str]],
    mutated_outputs: List[Optional[str]],
    in_spec: pytree.TreeSpec,
    out_spec: Optional[pytree.TreeSpec],
    state_dict: Dict[str, Any],
    constants: Dict[str, Any],
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
    unlifted_nodes = list(unlifted_name_to_node.values())

    gm.graph.lint()

    gm.graph._codegen = _get_codegen(in_spec, out_spec)
    gm.recompile()

    # Step 4: Find state references in HigherOrderOps and recursively
    # fix them.
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target.__name__ == "cond":
            pred, true_graph, false_graph, operands = node.args
            true_gm = getattr(gm, true_graph.name)
            false_gm = getattr(gm, false_graph.name)

            submod_lifted_inputs1, real_operands1 = _unlift_submod_inputs(
                true_gm, operands, gm, unlifted_nodes
            )
            submod_lifted_inputs2, real_operands2 = _unlift_submod_inputs(
                false_gm, operands, gm, unlifted_nodes
            )
            assert submod_lifted_inputs1 == submod_lifted_inputs2
            assert real_operands1 == real_operands2

            node.args = (pred, true_graph, false_graph, real_operands1)

            _, in_spec = pytree.tree_flatten(real_operands1)

            # Currently HOO submodules do not allow mutations, so we
            # will not need to handle this.
            output_node = None
            for node in gm.graph.nodes:
                if node.op == "output":
                    output_node = node
                    break
            assert output_node is not None
            outputs = pytree.tree_flatten(output_node.args)[0]
            mutated_outputs = [None for _ in range(len(outputs))]

            _unlift(
                true_gm,
                submod_lifted_inputs1,
                mutated_outputs,
                in_spec,
                None,
                state_dict,
                constants,
            )
            _unlift(
                false_gm,
                submod_lifted_inputs1,
                mutated_outputs,
                in_spec,
                None,
                state_dict,
                constants,
            )

        elif node.op == "call_function" and node.target.__name__ == "map_impl":
            body_graph, mapped_args, operands = node.args
            body_gm = getattr(gm, body_graph.name)

            def _find_real_operands(operands):
                submod_lifted_inputs = []
                real_operands = []
                for ix, operand in enumerate(operands):
                    if operand.target in lifted_inputs:
                        submod_lifted_inputs.append(operand.target)
                        if operand.target in state_dict:
                            value = state_dict[operand.target]
                        elif operand.target in constants:
                            value = constants[operand.target]
                        else:
                            raise RuntimeError(
                                "Unable to find value for ", operand.target
                            )

                        body_gm.register_buffer(operand.target, value)
                    else:
                        submod_lifted_inputs.append(None)
                        real_operands.append(operand)

                return submod_lifted_inputs, real_operands

            lifted_args, real_mapped_args = _unlift_submod_inputs(
                body_gm, mapped_args, gm, unlifted_nodes
            )
            lifted_operands, real_mapped_operands = _unlift_submod_inputs(
                body_gm, operands, gm, unlifted_nodes
            )

            node.args = (body_graph, real_mapped_args, real_mapped_operands)
            submod_lifted_inputs = lifted_args + lifted_operands

            _, in_spec = pytree.tree_flatten(real_mapped_args + real_mapped_operands)

            # Currently HOO submodules do not allow mutations, so we
            # will not need to handle this.
            output_node = None
            for node in gm.graph.nodes:
                if node.op == "output":
                    output_node = node
                    break
            assert output_node is not None
            outputs = pytree.tree_flatten(output_node.args)[0]
            mutated_outputs = [None for _ in range(len(outputs))]

            _unlift(
                body_gm,
                submod_lifted_inputs,
                mutated_outputs,
                in_spec,
                None,
                state_dict,
                constants,
            )

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
    for name, value in state_dict.items():
        if name in graph_signature.buffers:
            new_gm.register_buffer(name.replace(".", "_"), value)
        if name in graph_signature.parameters:
            new_gm.register_parameter(name.replace(".", "_"), value)

    for name, value in constants.items():
        setattr(new_gm, name.replace(".", "_"), value)


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
        self.range_constraints = range_constraints or []


def _create_stateful_graph_module(
    plain_graph_module: torch.fx.GraphModule,
    range_constraints,
):
    stateful_gm = _StatefulGraphModule._create(
        plain_graph_module,
        plain_graph_module.graph,
        range_constraints=range_constraints,
    )
    stateful_gm.register_forward_pre_hook(
        _check_input_constraints_pre_hook, with_kwargs=True
    )
    return stateful_gm


def _unlift_exported_program_lifted_states(ep: ExportedProgram) -> torch.nn.Module:
    new_gm = copy.deepcopy(ep.graph_module)
    _register_attrs_to_new_gm(new_gm, ep.graph_signature, ep.state_dict, ep.constants)

    lifted_inputs: List[Optional[str]] = [
        in_spec.target
        if in_spec.kind
        in (
            InputKind.BUFFER,
            InputKind.CONSTANT_TENSOR,
            InputKind.PARAMETER,
            InputKind.CUSTOM_OBJ,
        )
        else None
        for in_spec in ep.graph_signature.input_specs
    ]

    mutated_outputs: List[Optional[str]] = [
        out_spec.target
        if out_spec.kind in (OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION)
        else None
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
    )
    unlift_gm = _create_stateful_graph_module(new_gm, ep.range_constraints)
    unlift_gm.meta.update(ep.graph_module.meta)
    return unlift_gm
