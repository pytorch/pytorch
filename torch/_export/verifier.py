import inspect
import math
import operator
from collections.abc import Iterable
from typing import Any, Dict, final, List, Optional, Tuple, Type

import torch
from torch._ops import HigherOrderOperator, OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import (
    CustomObjArgument,
    InputKind,
    SymIntArgument,
    TensorArgument,
)
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import SymBool, SymFloat, SymInt


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
        elif isinstance(val, (FakeTensor, torch.Tensor)):  # TODO(zhxchen17) Remove Tensor.
            return True
        elif isinstance(val, (SymInt, SymFloat, SymBool)):
            return True
        elif isinstance(val, CustomObjArgument):
            return True
        elif isinstance(val, Iterable):
            return all(_check_correct_val(x) for x in val)
        return False

    def _no_returns(op):
        if not isinstance(op, OpOverload):
            return False
        return len(op._schema.returns) == 0

    if "val" not in node.meta:
        if node.op == "call_function" and _no_returns(node.target):
            return
        raise SpecViolationError(f"Node.meta {node.name} is missing val field.")

    val = node.meta["val"]
    if not _check_correct_val(val):
        raise SpecViolationError(f"Node.meta {node.name} has invalid val field {val}")


class _VerifierMeta(type):
    _registry: Dict[str, Type['Verifier']] = {}

    def __new__(metacls, name, bases, attrs):
        if bases:
            if "check" in attrs or "_check_graph_module" in attrs:
                raise SyntaxError("Overriding method check is not allowed.")
            assert "dialect" in attrs and attrs["dialect"] != "ATEN"
        else:
            assert "check" in attrs
            assert "_check_graph_module" in attrs
            assert attrs["dialect"] == "ATEN"

        assert isinstance(attrs["dialect"], str)
        ret = type.__new__(metacls, name, bases, attrs)
        metacls._registry[attrs["dialect"]] = ret  # type: ignore[assignment]
        return ret

def getattr_recursive(obj: Any, target: str) -> Any:
    target_atoms = target.split('.')
    attr_itr = obj
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


class Verifier(metaclass=_VerifierMeta):
    dialect = "ATEN"

    def allowed_builtin_ops(self) -> List:
        return [
            operator.getitem,
            operator.add,
            operator.mul,
            operator.sub,
            operator.truediv,
            operator.ge,
            operator.le,
            operator.gt,
            operator.lt,
            operator.eq,
            operator.ne,
            operator.floordiv,
            operator.mod,
            operator.and_,
            operator.or_,
            operator.not_,
            operator.pow,
            operator.neg,
            operator.abs,
            math.ceil,
            math.floor,
        ]

    def allowed_op_types(self) -> Tuple[Type[Any], ...]:
        return (OpOverload, HigherOrderOperator)

    def allowed_getattr_types(self) -> Tuple[Type[Any], ...]:
        return (torch.fx.GraphModule,)

    def check_valid_op(self, op):
        pass

    def check_additional(self, gm: GraphModule) -> None:
        """
        Additional checks that are specific to some dialects.
        """
        pass

    @final
    def check(self, ep: ExportedProgram) -> None:
        self._check_graph_module(ep.graph_module)
        _verify_exported_program_signature(ep)

    @final
    def _check_graph_module(self, gm: torch.fx.GraphModule) -> None:
        def _allowed_getattr_types() -> Tuple[Type[Any], ...]:
            ret = self.allowed_getattr_types()
            assert not any(t is object for t in ret)
            return ret

        def _check_valid_op(op) -> None:
            def _allowed_builtin_ops() -> List:
                ret = self.allowed_builtin_ops()
                assert all(inspect.isbuiltin(op) for op in ret)
                return ret

            def _allowed_op_types() -> Tuple[Type[Any], ...]:
                ret = self.allowed_op_types()
                assert not any(t is object for t in ret)
                return ret

            # TODO Remove this allowlist.
            _allowed_torch_functions = (
                torch.autograd.grad_mode.set_grad_enabled,
                torch.sym_int,
                torch.sym_ite,
                torch.sym_max,
                torch.sym_min,
                torch.sym_not,
                torch.sym_sqrt,
                # TODO (tmanlaibaatar)
                # Predispatch export is able to contain autograd ops.
                # These will be modeled as HOO later
                torch._C._set_grad_enabled

            )

            if not isinstance(op, _allowed_op_types()):
                if op not in _allowed_builtin_ops() and op not in _allowed_torch_functions:
                    raise SpecViolationError(
                        f"Operator '{op}' is not an allowed operator type: {_allowed_op_types()}\n"
                        f"Valid builtin ops: {_allowed_builtin_ops()}"
                        f"Valid torch functions: {_allowed_torch_functions}"
                    )

            if isinstance(op, OpOverload):
                # All ops functional
                if not is_functional(op):
                    raise SpecViolationError(
                        f"operator '{op}' is not functional"
                    )
            self.check_valid_op(op)

        for mod in gm.modules():
            if not isinstance(mod, torch.fx.GraphModule):
                continue

            mod.graph.lint()
            for node in mod.graph.nodes:
                # TODO(T140410192): should have fake tensor for all dialects
                if node.op in {"call_module", "call_method"}:
                    raise SpecViolationError(
                        f"call_module is not valid: got a class '{node.target}' ",
                    )

                elif node.op == "call_function":
                    _check_val(node)

                    _check_valid_op(node.target)

                elif node.op == "get_attr":
                    if not isinstance(node.target, str):
                        raise SpecViolationError(
                            f"Expected get_attr target to be string, but got {type(node.target)}"
                        )

                    attr = getattr_recursive(mod, node.target)
                    if isinstance(attr, torch.nn.Module):
                        def _is_type(name, ty):
                            return isinstance(getattr(attr, name, None), ty)
                        if type(attr).__name__ == "LoweredBackendModule":
                            if _is_type("backend_id", str) \
                                    and _is_type("processed_bytes", bytes) \
                                    and _is_type("compile_specs", list) \
                                    and hasattr(attr, "original_module"):
                                continue
                            else:
                                backend_id = getattr(attr, "backend_id", None)
                                processed_bytes = getattr(attr, "processed_bytes", None)
                                compile_specs = getattr(attr, "compile_specs", None)
                                raise SpecViolationError(
                                    f"Invalid get_attr type {type(attr)}. \n"
                                    f"LoweredBackendModule fields: "
                                    f"backend_id(str) : {type(backend_id)}, "
                                    f"processed_bytes(bytes) : {type(processed_bytes)}, "
                                    f"compile_specs(list) : {type(compile_specs)}"
                                )

                    if not isinstance(attr, _allowed_getattr_types()):
                        raise SpecViolationError(
                            f"Invalid get_attr type {type(attr)}. \n"
                            f"Valid get_attr types: {_allowed_getattr_types()}"
                        )


                elif node.op == "placeholder":
                    _check_val(node)
                # TODO(zhxchen17)
                # elif node.op == "output":
                #     _check_flattened_outputs()

        self.check_additional(gm)


def _verify_exported_program_signature(exported_program) -> None:
    # Check ExportedProgram signature matches
    gs = exported_program.graph_signature

    # Check every node in the signature exists in the graph
    input_node_names = [node.name for node in exported_program.graph.nodes if node.op == "placeholder"]

    if len(input_node_names) != len(gs.input_specs):
        raise SpecViolationError(
            f"Number of graph inputs ({len(input_node_names)}) "
            f"does not match number of inputs in the graph signature ({len(gs.user_inputs)})"
        )

    for input_spec, node in zip(gs.input_specs, input_node_names):
        if isinstance(input_spec.arg, (TensorArgument, SymIntArgument)):
            if input_spec.arg.name != node:
                raise SpecViolationError(
                    f"Input spec name {input_spec.arg.name} does not match node name {node}"
                )

        if input_spec.kind == InputKind.USER_INPUT:
            continue

        elif input_spec.kind == InputKind.PARAMETER:
            if not isinstance(input_spec.arg, TensorArgument):
                raise SpecViolationError(
                    f"Parameter {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead."
                )
            if input_spec.target is None:
                raise SpecViolationError(
                    f"InputSpec for {input_spec.name} has no target."
                )

            param = input_spec.target
            if param not in exported_program.state_dict:
                raise SpecViolationError(
                    f"Parameter {param} is not in the state dict."
                )

            if not isinstance(exported_program.state_dict[param], torch.nn.Parameter):
                raise SpecViolationError(
                    f"State dict entry for parameter {param} is not an instance of torch.nn.Parameter."
                )

        elif input_spec.kind == InputKind.BUFFER:
            if not isinstance(input_spec.arg, TensorArgument):
                raise SpecViolationError(
                    f"Buffer {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead."
                )
            if input_spec.target is None:
                raise SpecViolationError(
                    f"InputSpec for {input_spec.name} has no target."
                )

            buffer = input_spec.target
            if input_spec.persistent is None:
                raise SpecViolationError(
                    f"Buffer {buffer} is missing a persistence flag"
                )

            if input_spec.persistent is True and buffer not in exported_program.state_dict:
                raise SpecViolationError(
                    f"Buffer {buffer} is not in the state dict."
                )

            if input_spec.persistent is False and buffer in exported_program.state_dict:
                raise SpecViolationError(
                    f"Non-persistent buffer {buffer} is in the state dict, it should not be."
                )
        elif input_spec.kind == InputKind.CONSTANT_TENSOR:
            if not isinstance(input_spec.arg, TensorArgument):
                raise SpecViolationError(
                    f"Constant tensor {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead."
                )
            if input_spec.target is None:
                raise SpecViolationError(
                    f"InputSpec for {input_spec.name} has no target."
                )

            tensor_const = input_spec.target
            if tensor_const not in exported_program.constants:
                raise SpecViolationError(
                    f"Constant tensor {tensor_const} is not in the constants dictionary."
                )
        elif input_spec.kind == InputKind.CUSTOM_OBJ:
            if not isinstance(input_spec.arg, CustomObjArgument):
                raise SpecViolationError(
                    f"Custom object {input_spec.name} is not a custom object argument. Found {input_spec.arg} instead."
                )
            if input_spec.target is None:
                raise SpecViolationError(
                    f"InputSpec for {input_spec.name} has no target."
                )

            custom_obj = input_spec.target
            if custom_obj not in exported_program.constants:
                raise SpecViolationError(
                    f"Custom object {custom_obj} is not in the constants dictionary."
                )
        elif input_spec.kind == InputKind.TOKEN:
            if not isinstance(input_spec.arg, TensorArgument):
                raise SpecViolationError(
                    f"Constant tensor {input_spec.name} is not a tensor argument. Found {input_spec.arg} instead."
                )
        else:
            raise SpecViolationError(
                f"Unknown InputKind {input_spec.kind}."
            )

    # Check outputs
    output_node = list(exported_program.graph.nodes)[-1]
    assert output_node.op == "output"
    output_nodes = [
        arg.name if isinstance(arg, torch.fx.Node) else arg
        for arg in output_node.args[0]
    ]

    if len(output_nodes) != len(gs.output_specs):
        raise SpecViolationError(
            f"Number of output nodes {len(output_nodes)} is different "
            "Than the number of outputs specified by the graph signature: \n"
            f"Number of mutated buffers: {len(gs.buffers_to_mutate)}. \n"
            f"Number of user outputs: {len(gs.user_outputs)}. \n"
        )

    num_tokens = len(gs.output_tokens)
    end = len(gs.buffers_to_mutate) + len(gs.user_inputs_to_mutate) + num_tokens
    mutate_nodes: List[str] = output_nodes[num_tokens:end]
    user_output_nodes = output_nodes[end:end + len(gs.user_outputs)]

    for mutation_node in mutate_nodes:
        if mutation_node in gs.buffers_to_mutate:
            if gs.buffers_to_mutate[mutation_node] not in gs.buffers:
                raise SpecViolationError(
                    f"Buffer output {mutation_node} does not point to a buffer that exists. \n"
                    f"Dict of buffers that are mutated, in order: {gs.buffers_to_mutate} \n"
                    f"Buffer nodes available: {gs.buffers} \n"
                )
        elif mutation_node in gs.user_inputs_to_mutate:
            if gs.user_inputs_to_mutate[mutation_node] not in gs.user_inputs:
                raise SpecViolationError(
                    f"User input output {mutation_node} does not point to a user input that exists. \n"
                    f"Dict of user inputs that are mutated, in order: {gs.user_inputs_to_mutate} \n"
                    f"User input nodes available: {gs.user_inputs} \n")
        else:
            raise SpecViolationError(
                f"Mutation node {mutation_node} is neither a buffer nor a user input. "
                f"Buffers to mutate: {gs.buffers_to_mutate}, User inputs to mutate: {gs.user_inputs_to_mutate}"
            )

    for user_output_node, user_output_name in zip(user_output_nodes, gs.user_outputs):
        if user_output_node != user_output_name:
            raise SpecViolationError(
                f"User output {user_output_node} is not in the correct "
                "order or is not found in the "
                f"exported program's user_output list: {gs.user_outputs}. "
            )


def load_verifier(dialect: str) -> Optional[Type[Verifier]]:
    if dialect == "ATEN":
        return _VerifierMeta._registry.get(dialect)
    return _VerifierMeta._registry[dialect]
