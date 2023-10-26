import operator
from collections.abc import Iterable
from typing import Any, final, List, Set, Tuple, Type

import torch
from torch._ops import HigherOrderOperator, OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import SymBool, SymFloat, SymInt


PRESERVED_META_KEYS: Set[str] = {
    "val",
    "stack_trace",
    "source_fn_stack",
}


class SpecViolationError(Exception):
    pass


def is_functional(op: OpOverload) -> bool:
    return not op._schema.is_mutable


def _check_has_fake_tensor(node: torch.fx.Node) -> None:
    # TODO(angelayi): remove this in favor of _check_val
    return _check_val(node)


def _check_val(node: torch.fx.Node) -> None:
    def _check_correct_val(val):
        if val is None:
            return True
        elif isinstance(val, (int, bool, str, float)):
            return True
        elif isinstance(val, (torch.memory_format, torch.dtype, torch.device, torch.layout)):
            return True
        elif isinstance(val, (FakeTensor, torch.Tensor)):
            return True
        elif isinstance(val, (SymInt, SymFloat, SymBool)):
            return True
        elif isinstance(val, Iterable):
            return all(_check_correct_val(x) for x in val)
        return False

    if "val" not in node.meta:
        raise SpecViolationError(f"Node.meta {node.name} is missing val field.")

    val = node.meta["val"]
    if not _check_correct_val(val):
        raise SpecViolationError(f"Node.meta {node.name} has invalid val field {val}")


class Verifier:
    def __call__(self, gm: GraphModule) -> None:
        self.check_valid(gm)

    def allowed_builtin_ops(self) -> List:
        return [operator.getitem]

    def allowed_op_types(self) -> Tuple[Type[Any], ...]:
        return (OpOverload, HigherOrderOperator)

    def check_valid_op(self, op) -> None:
        if op not in self.allowed_builtin_ops():
            if not isinstance(op, self.allowed_op_types()):
                raise SpecViolationError(
                    f"Operator '{op}' is not an allowed operator type.\n"
                    f"Valid op types: {self.allowed_builtin_ops}"
                )

        if isinstance(op, OpOverload):
            # All ops functional
            if not is_functional(op):
                raise SpecViolationError(
                    f"operator '{op}' is not functional"
                )

    def allowed_getattr_types(self) -> Tuple[Type[Any], ...]:
        return (torch.fx.GraphModule,)

    def check_additional(self, gm: GraphModule) -> None:
        """
        Additional checks that are specific to some dialects.
        """
        pass

    @final
    def check_valid(self, gm: GraphModule) -> None:  # noqa: C901
        gm.graph.lint()

        if object in self.allowed_op_types():
            raise SpecViolationError(
                "'object' is too generic to be in the list of allowed op types"
            )
        if object in self.allowed_getattr_types():
            raise SpecViolationError(
                "'object' is too generic to be in the list of allowed getattr types"
            )

        for mod in gm.modules():
            if not isinstance(mod, torch.fx.GraphModule):
                continue

            for node in mod.graph.nodes:
                # TODO(T140410192): should have fake tensor for all dialects
                if node.op in {"call_module", "call_method"}:
                    raise SpecViolationError(
                        f"call_module is not valid: got a class '{node.target}' ",
                    )

                elif node.op == "call_function":
                    _check_val(node)

                    self.check_valid_op(node.target)

                    if isinstance(node.target, OpOverload):
                        # Check preserved metadata
                        for meta in PRESERVED_META_KEYS:
                            if node.meta.get(meta, None) is None:
                                raise SpecViolationError(
                                    f"node {node} is missing metadata {meta}"
                                )

                elif node.op == "get_attr":
                    if not isinstance(node.target, str):
                        raise SpecViolationError(
                            f"Expected get_attr target to be string, but got {type(node.target)}"
                        )

                    attr = getattr(mod, node.target)
                    if not isinstance(attr, self.allowed_getattr_types()):
                        raise SpecViolationError(
                            f"Invalid get_attr type {type(attr)}. \n"
                            f"Valid get_attr types: {self.allowed_getattr_types}"
                        )


                elif node.op == "placeholder":
                    _check_val(node)

        self.check_additional(gm)

    def is_valid(self, gm: GraphModule) -> bool:
        try:
            self.check_valid(gm)
            return True
        except SpecViolationError:
            return False


class ATenDialectVerifier(Verifier):
    def check_valid_op(self, op) -> None:

        super().check_valid_op(op)

        if isinstance(op, OpOverload):
            if (
                torch.Tag.core not in op.tags
                and torch.Tag.view_copy not in op.tags
            ):
                # NOTE(qihan): whether view_copy operators are marked as canonical is still under
                #            discussion.
                raise SpecViolationError(
                    f"Operator {op.__module__}.{op.__name__} is not Aten Canonical."
                )


def verify_exported_program_signature(exported_program) -> None:
    # Check ExportedProgram signature matches
    gs = exported_program.graph_signature

    bs_grad_to_param = {}
    bs_grad_to_user_inputs = {}
    if gs.backward_signature is not None:
        bs_grad_to_param = gs.backward_signature.gradients_to_parameters
        bs_grad_to_user_inputs = gs.backward_signature.gradients_to_user_inputs

    # Check every node in the signature exists in the graph
    input_node_names = [node.name for node in exported_program.graph.nodes if node.op == "placeholder"]
    for node in exported_program.graph.nodes:
        if node.op != "placeholder":
            break
        input_node_names.append(node.name)
    output_node = list(exported_program.graph.nodes)[-1]
    assert output_node.op == "output"
    output_node_names = [node.name for node in output_node.args[0]]

    def check_exists(node_list, container):
        for node in node_list:
            if node not in container:
                raise SpecViolationError(
                    f"Node {node} found in the signature's is not in the graph."
                )
    check_exists(gs.user_inputs, input_node_names)
    check_exists(gs.user_outputs, output_node_names)
    check_exists(gs.inputs_to_parameters.keys(), input_node_names)
    check_exists(gs.inputs_to_parameters.values(), gs.parameters)
    check_exists(gs.inputs_to_buffers.keys(), input_node_names)
    check_exists(gs.inputs_to_buffers.values(), gs.buffers)
    check_exists(gs.buffers_to_mutate.keys(), output_node_names)
    check_exists(gs.buffers_to_mutate.values(), gs.buffers)

    check_exists(bs_grad_to_param.keys(), output_node_names)
    check_exists(bs_grad_to_param.values(), gs.parameters)
    check_exists(bs_grad_to_user_inputs.keys(), output_node_names)
    check_exists(bs_grad_to_user_inputs.values(), gs.user_inputs)

    # Check parameters
    for param in gs.parameters:
        if param not in exported_program.state_dict:
            raise SpecViolationError(
                f"Parameter {param} is not in the state dict."
            )

        if not isinstance(exported_program.state_dict[param], torch.nn.Parameter):
            raise SpecViolationError(
                f"State dict entry for parameter {param} is not an instance of torch.nn.Parameter."
            )

    # Check buffers
    for buffer in gs.buffers:
        if buffer not in exported_program.state_dict:
            raise SpecViolationError(
                f"Buffer {buffer} is not in the state dict."
            )

    # Check inputs
    placeholder_nodes = [n.name for n in exported_program.graph.nodes if n.op == "placeholder"]
    total_gs_placeholders = len(gs.inputs_to_parameters) + len(gs.inputs_to_buffers) + len(gs.user_inputs)
    if len(placeholder_nodes) != total_gs_placeholders:
        raise SpecViolationError(
            f"Number of placeholders nodes {len(placeholder_nodes)} is different "
            "Than the number of inputs specified by the graph signature: \n"
            f"Number of parameters: {len(gs.inputs_to_parameters)}. \n"
            f"Number of buffers: {len(gs.inputs_to_buffers)}. \n"
            f"Number of user inputs: {len(gs.user_inputs)}. \n"
        )

    parameter_nodes = placeholder_nodes[:len(gs.parameters)]
    buffer_nodes = placeholder_nodes[len(gs.parameters):len(gs.parameters) + len(gs.buffers)]
    user_input_nodes = placeholder_nodes[len(gs.parameters) + len(gs.buffers):]

    for param_node, param_name in zip(parameter_nodes, gs.parameters):
        if (
            param_node not in gs.inputs_to_parameters or
            gs.inputs_to_parameters[param_node] != param_name
        ):
            raise SpecViolationError(
                f"Parameter input {param_node} is not in the correct "
                "order or is not found in the exported program's parameter list. \n"
                f"List of parameters, in order: {gs.parameters} \n"
                f"Parameter node to parameter name mapping: {gs.inputs_to_parameters} \n"
            )

    for buffer_node, buffer_name in zip(buffer_nodes, gs.buffers):
        if (
            buffer_node not in gs.inputs_to_buffers or
            gs.inputs_to_buffers[buffer_node] != buffer_name
        ):
            raise SpecViolationError(
                f"Buffer input {buffer_node} is not in the correct "
                "order or is not found in the exported program's buffer list. \n"
                f"List of buffers, in order: {gs.buffers} \n"
                f"Buffer node to buffer name mapping: {gs.inputs_to_buffers} \n"
            )

    for user_input_node, user_input_name in zip(user_input_nodes, gs.user_inputs):
        if user_input_node != user_input_name:
            raise SpecViolationError(
                f"User input {user_input_node} is not in the correct "
                "order or is not found in the "
                f"exported program's user_input list: {gs.user_input}. "
            )

    # Check outputs
    output_node = list(exported_program.graph.nodes)[-1]
    assert output_node.op == "output"
    output_nodes = [arg.name for arg in output_node.args[0]]

    total_gs_outputs = (
        len(gs.buffers_to_mutate) +
        len(gs.user_outputs) +
        len(bs_grad_to_param) +
        len(bs_grad_to_user_inputs)
    )
    if len(output_nodes) != total_gs_outputs:
        raise SpecViolationError(
            f"Number of output nodes {len(output_nodes)} is different "
            "Than the number of outputs specified by the graph signature: \n"
            f"Number of mutated buffers: {len(gs.buffers_to_mutate)}. \n"
            f"Number of user outputs: {len(gs.user_outputs)}. \n"
        )

    buffer_mutate_nodes = output_nodes[:len(gs.buffers_to_mutate)]
    user_output_nodes = output_nodes[len(gs.buffers_to_mutate):len(gs.user_outputs) + len(gs.buffers_to_mutate)]

    for buffer_node in buffer_mutate_nodes:
        if (
            buffer_node not in gs.buffers_to_mutate or
            gs.buffers_to_mutate[buffer_node] not in gs.buffers
        ):
            raise SpecViolationError(
                f"Buffer output {buffer_node} is not in buffer mutation dictionary "
                "or, it does not point to a buffer that exists. \n"
                f"Dict of buffers that are mutated, in order: {gs.buffers_to_mutate} \n"
                f"Buffer nodes available: {gs.buffers} \n"
            )

    for user_output_node, user_output_name in zip(user_output_nodes, gs.user_outputs):
        if user_output_node != user_output_name:
            raise SpecViolationError(
                f"User output {user_output_node} is not in the correct "
                "order or is not found in the "
                f"exported program's user_output list: {gs.user_output}. "
            )
