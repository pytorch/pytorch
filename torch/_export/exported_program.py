import copy
from collections import defaultdict
import dataclasses
from typing import Dict, List, Optional, Tuple
import warnings

import sympy

import torch
import torch.fx

import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import SymInt
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo

from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
    InputDim,
    RangeConstraint,
)


# TODO(ycao): This is added to avoid breaking existing code temporarily.
# Remove when migration is done.
from torch.export import (
    ArgumentKind,
    ArgumentSpec,
    ExportBackwardSignature,
    ExportGraphSignature,
    ExportedProgram,
    ModuleCallEntry,
    ModuleCallSignature,
)


__all__ = [
    "ArgumentKind",
    "ArgumentSpec",
    "ExportBackwardSignature",
    "ExportGraphSignature",
    "ExportedProgram",
    "ModuleCallEntry",
    "ModuleCallSignature",
]


# Information to maintain user calling/returning specs
@dataclasses.dataclass
class CallSpec:
    in_spec: Optional[pytree.TreeSpec]
    out_spec: Optional[pytree.TreeSpec]


def _unlift(gm, inp_pos_to_param_buffer_name, in_spec, out_spec, state_dict, buffers_to_mutate, user_outputs):
    count = 0
    buffer_name_to_node = {}
    # Step 1: make lifted params as get_attr
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if count in inp_pos_to_param_buffer_name:
                with gm.graph.inserting_after(node):
                    getattr_node = gm.graph.get_attr(
                        inp_pos_to_param_buffer_name[count]
                    )
                    node.replace_all_uses_with(getattr_node)
                    metadata = node.meta
                    gm.graph.erase_node(node)
                    getattr_node.meta = metadata
                    buffer_name_to_node[inp_pos_to_param_buffer_name[count]] = getattr_node

            count += 1
        # Step 2: Find the all the buffers that were mutated and update them
        if node.op == "output":
            user_output_nodes = []
            for return_node in node.all_input_nodes:
                return_node_name = return_node.name
                # we found a param/buffer mutation
                if return_node_name in buffers_to_mutate:
                    buffer_node_name = buffers_to_mutate[return_node_name]
                    assert buffer_node_name in buffer_name_to_node
                    buffer_node = buffer_name_to_node[buffer_node_name]
                    with gm.graph.inserting_before(node):
                        buffer_update_node = gm.graph.call_function(
                            torch.ops.aten.copy_.default, (buffer_node, return_node)
                        )
                else:
                    user_output_nodes.append(return_node)
            with gm.graph.inserting_before(node):
                # Only return user outputs
                new_output = gm.graph.output(tuple(user_output_nodes))
                node.replace_all_uses_with(new_output)
                gm.graph.erase_node(node)

    # Step 3: Fix the input/output of the graph now that we deleted
    # some args.
    gm.graph.lint()
    names = [f"arg_{i}" for i in range(len(in_spec.children_specs))]
    gm.graph._codegen = _PyTreeCodeGen(
        _PyTreeInfo(
            names,
            in_spec,
            out_spec,
        )
    )
    gm.recompile()

    # Step 4: Find state references in HigherOrderOps and recursively
    # fix them.
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.cond:
            pred, true_graph, false_graph, operands = node.args
            true_gm = getattr(gm, true_graph.name)
            false_gm = getattr(gm, false_graph.name)
            inp_pos_to_param_buffer_name_for_submod = {}
            real_operands = []
            for ix, operand in enumerate(operands):
                if operand.target in inp_pos_to_param_buffer_name.values():
                    inp_pos_to_param_buffer_name_for_submod[ix] = operand.target
                    true_gm.register_buffer(operand.target, state_dict[operand.target])
                    false_gm.register_buffer(operand.target, state_dict[operand.target])
                else:
                    real_operands.append(operand)
            node.args = (pred, true_graph, false_graph, real_operands)

            _, in_spec = pytree.tree_flatten(real_operands)

            _unlift(
                true_gm,
                inp_pos_to_param_buffer_name_for_submod,
                in_spec,
                None,
                state_dict,
                buffers_to_mutate,
                user_outputs,
            )
            _unlift(
                false_gm,
                inp_pos_to_param_buffer_name_for_submod,
                in_spec,
                None,
                state_dict,
                buffers_to_mutate,
                user_outputs,
            )
        if node.op == "call_function" and node.target.__name__ == "map_impl":
            body_graph, num_mapped, *operands = node.args
            body_gm = getattr(gm, body_graph.name)
            inp_pos_to_buffer_name_for_submod = {}
            real_operands = []
            for ix, operand in enumerate(operands):
                if operand.target in inp_pos_to_param_buffer_name.values():
                    inp_pos_to_buffer_name_for_submod[ix] = operand.target
                    body_gm.register_buffer(operand.target, state_dict[operand.target])
                else:
                    real_operands.append(operand)
            node.args = (body_graph, num_mapped, *real_operands)

            _, in_spec = pytree.tree_flatten(real_operands)

            _unlift(
                body_gm,
                inp_pos_to_buffer_name_for_submod,
                in_spec,
                None,
                state_dict,
                buffers_to_mutate,
                user_outputs,
            )
    gm.graph.lint()
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def unlift_exported_program_lifted_states(ep: torch.export.ExportedProgram) -> torch.nn.Module:
    new_gm = copy.deepcopy(ep.graph_module)

    # TODO Fix the period in params/buffers names later
    # maybe a pass to replace graph signature with fixed names
    param_buffer_name_to_corrected_name = {}

    for name, value in ep.state_dict.items():
        if name in ep.graph_signature.buffers:
            if "." in name:
                new_gm.register_buffer(name.replace(".", "_"), value)
                param_buffer_name_to_corrected_name[name] = name.replace(".", "_")
            else:
                new_gm.register_buffer(name, value)
        if name in ep.graph_signature.parameters:
            if "." in name:
                new_gm.register_parameter(name.replace(".", "_"), value)
                param_buffer_name_to_corrected_name[name] = name.replace(".", "_")
            else:
                new_gm.register_parameter(name, value)

    count = 0
    inp_pos_to_param_buffer_name = {}
    for node in new_gm.graph.nodes:
        if node.op == "placeholder":
            if node.name in ep.graph_signature.inputs_to_buffers:
                buffer_name = ep.graph_signature.inputs_to_buffers[node.name]
                if buffer_name in param_buffer_name_to_corrected_name:
                    inp_pos_to_param_buffer_name[
                        count
                    ] = param_buffer_name_to_corrected_name[buffer_name]
                else:
                    inp_pos_to_param_buffer_name[count] = buffer_name
            if node.name in ep.graph_signature.inputs_to_parameters:
                param_name = ep.graph_signature.inputs_to_parameters[node.name]
                if param_name in param_buffer_name_to_corrected_name:
                    inp_pos_to_param_buffer_name[
                        count
                    ] = param_buffer_name_to_corrected_name[param_name]
                else:
                    inp_pos_to_param_buffer_name[count] = param_name
            count += 1
    new_gm = _unlift(
        new_gm,
        inp_pos_to_param_buffer_name,
        ep.call_spec.in_spec,
        ep.call_spec.out_spec,
        ep.state_dict,
        ep.graph_signature.buffers_to_mutate,
        ep.graph_signature.user_outputs,
    )
    new_gm.meta.update(ep.graph_module.meta)
    return new_gm


def _create_graph_module_for_export(root, graph):
    try:
        gm = torch.fx.GraphModule(root, graph)
    except SyntaxError:
        # If custom objects stored in memory are being used in the graph,
        # the generated python code will result in a syntax error on the custom
        # object, since it is unable to parse the in-memory object. However
        # we can still run the graph eagerly through torch.fx.Interpreter,
        # so we will bypass this error.
        warnings.warn(
            "Unable to execute the generated python source code from "
            "the graph. The graph module will no longer be directly callable, "
            "but you can still run the ExportedProgram, and if needed, you can "
            "run the graph module eagerly using torch.fx.Interpreter.",
            stacklevel=2,
        )
        gm = torch.fx.GraphModule(root, torch.fx.Graph())
        gm._graph = graph

    return gm


def _process_constraints(
    graph_module: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    example_inputs: List[torch.Tensor],
) -> Tuple[Dict[sympy.Symbol, RangeConstraint], List[Tuple[InputDim, InputDim]]]:
    """
    Process the constraints stored in the graph module to return something more readable.

    Args:
        graph_module (torch.fx.GraphModule): GraphModule returned from
            dynamo.export, which contains the "input_shape_constraints" and
            "inline_constraints" metadata

        example_inputs: Flattened list of example inputs used to export the graph module

    Returns:
        range_constraints (Dict[sympy.Symbol, RangeConstraints]): Mapping of
            symbols (from SymInts) appearing in the fake tensors in
            node.meta["val"] to their range constraints, which are a tuple
            containing (lower, upper) constraints.

        equality_constraints (List[Tuple[InputDim, InputDim]]): List of tuples
            of (node, dim) to mark that these dimensions are equal.
    """
    input_shape_constraints = graph_module.meta.get("input_shape_constraints", [])
    inline_constraints = graph_module.meta.get("inline_constraints", [])
    num_params_buffer = len(graph_signature.buffers) + len(graph_signature.parameters)

    # Create dict mapping tensor_id to node names
    tensor_id_to_nodes: Dict[int, List[str]] = defaultdict(list)
    # Create dict mapping placeholder node names to their nodes
    placeholder_nodes: Dict[str, torch.fx.Node] = {}
    for i, node in enumerate(graph_module.graph.nodes):
        if node.op != "placeholder":
            # All placeholder nodes should be together in the beginning of the
            # graph
            break
        if i >= num_params_buffer:
            example_input = example_inputs[i - num_params_buffer]
            tensor_id_to_nodes[id(example_input)].append(node.name)
            placeholder_nodes[node.name] = node

    # Create list of (node name, dim) tuples to mark that they are equal
    equality_constraints: List[Tuple[InputDim, InputDim]] = []
    # Create dict mapping (node name, dim) a list of range (lower, upper)
    # constraints
    multi_range_constraints: Dict[InputDim, List[RangeConstraint]] = defaultdict(list)
    for constraint in input_shape_constraints:
        for node in tensor_id_to_nodes[constraint["t_id"]]:
            node_dim = InputDim(node, constraint["dim"])

            # Accumulate range constraints
            multi_range_constraints[node_dim].append(
                RangeConstraint(constraint["min"], constraint["max"])
            )

            # Accumulate equality constraints
            if shared := constraint.get("shared", None):
                for other_node in tensor_id_to_nodes[shared["t_id"]]:
                    other_node_dim = InputDim(other_node, shared["dim"])
                    equality_constraints.append((node_dim, other_node_dim))

    # Create dict mapping symbol to a singular range (lower, upper)
    range_constraints: Dict[sympy.Symbol, RangeConstraint] = {}

    # Add inline constraints to range_constraints
    for symbol, value_range in inline_constraints.items():
        range_constraints[symbol] = RangeConstraint(value_range.lower, value_range.upper)

    # Add input range constraints to range_constraintss
    for input_dim, multi_range_constraint in multi_range_constraints.items():  # type: ignore[assignment]
        # Simplify the range constraints into a single range constraint
        # Ex. ranges [2, 10] and [3, 11] would get merged to [3, 10]
        min_vals = [rc.min_val for rc in multi_range_constraint]
        max_vals = [rc.max_val for rc in multi_range_constraint]
        min_val = max(min_vals)
        max_val = min(max_vals)
        assert min_val <= max_val

        # Add input node range constraints
        val = placeholder_nodes[input_dim.input_name].meta["val"]
        assert isinstance(val, FakeTensor)
        symint = val.shape[input_dim.dim]
        assert isinstance(symint, SymInt)
        symbol = symint.node._expr
        range_constraints[symbol] = RangeConstraint(min_val, max_val)

    return range_constraints, equality_constraints

def combine_args_kwargs(args, kwargs):
    return (args, kwargs) if kwargs else args
