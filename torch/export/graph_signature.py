# mypy: allow-untyped-defs
import dataclasses
from collections.abc import Collection, Mapping
from enum import auto, Enum
from typing import TYPE_CHECKING, Union

from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import get_opaque_type_name, is_opaque_type
from torch._subclasses.fake_tensor import is_fake


if TYPE_CHECKING:
    import torch
    from torch._functorch._aot_autograd.schemas import GraphSignature

__all__ = [
    "ConstantArgument",
    "CustomObjArgument",
    "ExportBackwardSignature",
    "ExportGraphSignature",
    "InputKind",
    "InputSpec",
    "OutputKind",
    "OutputSpec",
    "SymIntArgument",
    "SymFloatArgument",
    "SymBoolArgument",
    "TensorArgument",
]


@dataclasses.dataclass
class TensorArgument:
    name: str


@dataclasses.dataclass
class TokenArgument:
    name: str


@dataclasses.dataclass
class SymIntArgument:
    name: str


@dataclasses.dataclass
class SymFloatArgument:
    name: str


@dataclasses.dataclass
class SymBoolArgument:
    name: str


@dataclasses.dataclass
class CustomObjArgument:
    name: str
    class_fqn: str
    fake_val: FakeScriptObject | None = None


@dataclasses.dataclass
class ConstantArgument:
    name: str
    value: int | float | bool | str | None


ArgumentSpec = Union[
    TensorArgument,
    SymIntArgument,
    SymFloatArgument,
    SymBoolArgument,
    ConstantArgument,
    CustomObjArgument,
    TokenArgument,
]


class InputKind(Enum):
    USER_INPUT = auto()
    PARAMETER = auto()
    BUFFER = auto()
    CONSTANT_TENSOR = auto()
    CUSTOM_OBJ = auto()
    TOKEN = auto()


@dataclasses.dataclass
class InputSpec:
    kind: InputKind
    arg: ArgumentSpec
    target: str | None
    persistent: bool | None = None

    def __post_init__(self):
        if self.kind == InputKind.BUFFER:
            if self.persistent is None:
                raise AssertionError("Failed to specify persistent flag on BUFFER.")
        if not isinstance(
            self.arg,
            (
                TensorArgument,
                SymIntArgument,
                SymFloatArgument,
                SymBoolArgument,
                ConstantArgument,
                CustomObjArgument,
                TokenArgument,
            ),
        ):
            raise AssertionError(f"expected valid arg type, got {type(self.arg)}")

    def __str__(self):
        target = "" if self.target is None else f" target='{self.target}'"
        persistent = "" if self.persistent is None else f" persistent={self.persistent}"
        return f"{str(self.arg.name)}: {str(self.kind.name)}{target}{persistent}"


class OutputKind(Enum):
    USER_OUTPUT = auto()
    LOSS_OUTPUT = auto()
    BUFFER_MUTATION = auto()
    PARAMETER_MUTATION = auto()
    GRADIENT_TO_PARAMETER = auto()
    GRADIENT_TO_USER_INPUT = auto()
    USER_INPUT_MUTATION = auto()
    TOKEN = auto()


@dataclasses.dataclass
class OutputSpec:
    kind: OutputKind
    arg: ArgumentSpec
    target: str | None

    def __post_init__(self):
        if not isinstance(
            self.arg,
            (
                TensorArgument,
                SymIntArgument,
                SymFloatArgument,
                SymBoolArgument,
                ConstantArgument,
                TokenArgument,
                CustomObjArgument,
            ),
        ):
            raise AssertionError(f"expected valid arg type, got {self.arg}")

    def __str__(self):
        target = "" if self.target is None else f" target='{self.target}'"
        return f"{str(self.arg.name)}: {str(self.kind.name)}{target}"


@dataclasses.dataclass
class ExportBackwardSignature:
    gradients_to_parameters: dict[str, str]
    gradients_to_user_inputs: dict[str, str]
    loss_output: str


@dataclasses.dataclass
class ExportGraphSignature:
    """
    :class:`ExportGraphSignature` models the input/output signature of Export Graph,
    which is a fx.Graph with stronger invariants guarantees.

    Export Graph is functional and does not access "states" like parameters
    or buffers within the graph via ``getattr`` nodes. Instead, :func:`export`
    guarantees that parameters, buffers, and constant tensors are lifted out of
    the graph as inputs.  Similarly, any mutations to buffers are not included
    in the graph either, instead the updated values of mutated buffers are
    modeled as additional outputs of Export Graph.

    The ordering of all inputs and outputs are::

        Inputs = [*parameters_buffers_constant_tensors, *flattened_user_inputs]
        Outputs = [*mutated_inputs, *flattened_user_outputs]

    e.g. If following module is exported::

        class CustomModule(nn.Module):
            def __init__(self) -> None:
                super(CustomModule, self).__init__()

                # Define a parameter
                self.my_parameter = nn.Parameter(torch.tensor(2.0))

                # Define two buffers
                self.register_buffer("my_buffer1", torch.tensor(3.0))
                self.register_buffer("my_buffer2", torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0)  # In-place addition

                return output


        mod = CustomModule()
        ep = torch.export.export(mod, (torch.tensor(1.0), torch.tensor(2.0)))

    Resulting Graph is non-functional::

        graph():
            %p_my_parameter : [num_users=1] = placeholder[target=p_my_parameter]
            %b_my_buffer1 : [num_users=1] = placeholder[target=b_my_buffer1]
            %b_my_buffer2 : [num_users=2] = placeholder[target=b_my_buffer2]
            %x1 : [num_users=1] = placeholder[target=x1]
            %x2 : [num_users=1] = placeholder[target=x2]
            %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x1, %p_my_parameter), kwargs = {})
            %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %b_my_buffer1), kwargs = {})
            %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%x2, %b_my_buffer2), kwargs = {})
            %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
            %add_ : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%b_my_buffer2, 1.0), kwargs = {})
            return (add_1,)

    Resulting ExportGraphSignature of the non-functional Graph would be::

        # inputs
        p_my_parameter: PARAMETER target='my_parameter'
        b_my_buffer1: BUFFER target='my_buffer1' persistent=True
        b_my_buffer2: BUFFER target='my_buffer2' persistent=True
        x1: USER_INPUT
        x2: USER_INPUT

        # outputs
        add_1: USER_OUTPUT

    To get a functional Graph, you can use :func:`run_decompositions`::

        mod = CustomModule()
        ep = torch.export.export(mod, (torch.tensor(1.0), torch.tensor(2.0)))
        ep = ep.run_decompositions()

    Resulting Graph is functional::

        graph():
            %p_my_parameter : [num_users=1] = placeholder[target=p_my_parameter]
            %b_my_buffer1 : [num_users=1] = placeholder[target=b_my_buffer1]
            %b_my_buffer2 : [num_users=2] = placeholder[target=b_my_buffer2]
            %x1 : [num_users=1] = placeholder[target=x1]
            %x2 : [num_users=1] = placeholder[target=x2]
            %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x1, %p_my_parameter), kwargs = {})
            %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %b_my_buffer1), kwargs = {})
            %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%x2, %b_my_buffer2), kwargs = {})
            %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
            %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%b_my_buffer2, 1.0), kwargs = {})
            return (add_2, add_1)

    Resulting ExportGraphSignature of the functional Graph would be::

        # inputs
        p_my_parameter: PARAMETER target='my_parameter'
        b_my_buffer1: BUFFER target='my_buffer1' persistent=True
        b_my_buffer2: BUFFER target='my_buffer2' persistent=True
        x1: USER_INPUT
        x2: USER_INPUT

        # outputs
        add_2: BUFFER_MUTATION target='my_buffer2'
        add_1: USER_OUTPUT

    """

    input_specs: list[InputSpec]
    output_specs: list[OutputSpec]

    # A list of parameters uniquely identified by mangled fully qualified name
    @property
    def parameters(self) -> Collection[str]:
        return tuple(
            s.target
            for s in self.input_specs
            if s.kind == InputKind.PARAMETER
            if isinstance(s.target, str)
        )

    # A list of buffers uniquely identified by mangled fully qualified name
    @property
    def buffers(self) -> Collection[str]:
        return tuple(
            s.target
            for s in self.input_specs
            if s.kind == InputKind.BUFFER
            if isinstance(s.target, str)
        )

    @property
    def non_persistent_buffers(self) -> Collection[str]:
        return tuple(
            s.target
            for s in self.input_specs
            if s.kind == InputKind.BUFFER
            if s.persistent is False
            if isinstance(s.target, str)
        )

    # A list of lifted constant tensors
    @property
    def lifted_tensor_constants(self) -> Collection[str]:
        return tuple(
            s.target
            for s in self.input_specs
            if s.kind == InputKind.CONSTANT_TENSOR
            if isinstance(s.target, str)
        )

    @property
    def lifted_custom_objs(self) -> Collection[str]:
        return tuple(
            s.target
            for s in self.input_specs
            if s.kind == InputKind.CUSTOM_OBJ
            if isinstance(s.target, str)
        )

    # Graph node names of pytree-flattened inputs of original program
    @property
    def user_inputs(self) -> Collection[int | float | bool | str | None]:
        user_inputs: list[int | float | bool | str | None] = []
        for s in self.input_specs:
            if s.kind != InputKind.USER_INPUT:
                continue

            if isinstance(
                s.arg,
                (
                    TensorArgument,
                    SymIntArgument,
                    SymFloatArgument,
                    SymBoolArgument,
                    CustomObjArgument,
                ),
            ):
                user_inputs.append(s.arg.name)
            elif isinstance(s.arg, ConstantArgument):
                user_inputs.append(s.arg.value)
            else:
                raise RuntimeError(f"{s.arg} is not a valid user inputs")
        return tuple(user_inputs)

    # Graph node names of pytree-flattened outputs of original program
    # For joint-graph purposes, will include the loss output.
    @property
    def user_outputs(self) -> Collection[int | float | bool | str | None]:
        user_outputs: list[int | float | bool | str | None] = []
        for s in self.output_specs:
            if s.kind not in [
                OutputKind.USER_OUTPUT,
                OutputKind.LOSS_OUTPUT,
            ]:
                continue

            if isinstance(
                s.arg,
                (TensorArgument, SymIntArgument, SymFloatArgument, SymBoolArgument),
            ):
                user_outputs.append(s.arg.name)
            elif isinstance(s.arg, ConstantArgument):
                user_outputs.append(s.arg.value)
            elif isinstance(s.arg, CustomObjArgument):
                user_outputs.append(s.arg.name)
            else:
                raise RuntimeError(f"{s.arg} is not a valid user output")
        return tuple(user_outputs)

    # A dictionary mapping graph input node names to parameters. If a graph input
    # name is found in this dictionary, it is guaranteed to be a lifted parameter.
    @property
    def inputs_to_parameters(self) -> Mapping[str, str]:
        return _immutable_dict(
            (s.arg.name, s.target)
            for s in self.input_specs
            if s.kind == InputKind.PARAMETER
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        )

    # A dictionary mapping graph input node names to buffers. If a graph input
    # name is found in this dictionary, it is guaranteed to be a lifted buffer.
    @property
    def inputs_to_buffers(self) -> Mapping[str, str]:
        return _immutable_dict(
            (s.arg.name, s.target)  # type: ignore[union-attr, misc]
            for s in self.input_specs
            if s.kind == InputKind.BUFFER
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        )

    # A dictionary mapping graph output node names to buffers that are mutated in the
    # original program. Buffers that are not mutated will not be found in this dictionary.
    @property
    def buffers_to_mutate(self) -> Mapping[str, str]:
        return _immutable_dict(
            (s.arg.name, s.target)
            for s in self.output_specs
            if s.kind == OutputKind.BUFFER_MUTATION
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        )

    @property
    def parameters_to_mutate(self) -> Mapping[str, str]:
        return _immutable_dict(
            (s.arg.name, s.target)
            for s in self.output_specs
            if s.kind == OutputKind.PARAMETER_MUTATION
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        )

    @property
    def user_inputs_to_mutate(self) -> Mapping[str, str]:
        return _immutable_dict(
            (s.arg.name, s.target)
            for s in self.output_specs
            if s.kind == OutputKind.USER_INPUT_MUTATION
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        )

    # A dictionary mapping graph input node names to lifted tensor constants.
    @property
    def inputs_to_lifted_tensor_constants(self) -> Mapping[str, str]:
        return _immutable_dict(
            (s.arg.name, s.target)
            for s in self.input_specs
            if s.kind == InputKind.CONSTANT_TENSOR
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        )

    @property
    def inputs_to_lifted_custom_objs(self) -> Mapping[str, str]:
        return _immutable_dict(
            (s.arg.name, s.target)
            for s in self.input_specs
            if s.kind == InputKind.CUSTOM_OBJ
            and isinstance(s.arg, CustomObjArgument)
            and isinstance(s.target, str)
        )

    @property
    def backward_signature(self) -> ExportBackwardSignature | None:
        loss_output = None
        gradients_to_parameters: dict[str, str] = {}
        gradients_to_user_inputs: dict[str, str] = {}
        for spec in self.output_specs:
            if spec.kind == OutputKind.LOSS_OUTPUT:
                if loss_output is not None:
                    raise AssertionError("multiple LOSS_OUTPUT specs found")
                if not isinstance(spec.arg, TensorArgument):
                    raise AssertionError(
                        f"expected TensorArgument for LOSS_OUTPUT, got {type(spec.arg)}"
                    )
                loss_output = spec.arg.name
            elif spec.kind == OutputKind.GRADIENT_TO_PARAMETER:
                if not isinstance(spec.target, str):
                    raise AssertionError(
                        f"expected str target for GRADIENT_TO_PARAMETER, got {type(spec.target)}"
                    )
                if not isinstance(spec.arg, TensorArgument):
                    raise AssertionError(
                        f"expected TensorArgument for GRADIENT_TO_PARAMETER, got {type(spec.arg)}"
                    )
                gradients_to_parameters[spec.arg.name] = spec.target
            elif spec.kind == OutputKind.GRADIENT_TO_USER_INPUT:
                if not isinstance(spec.target, str):
                    raise AssertionError(
                        f"expected str target for GRADIENT_TO_USER_INPUT, got {type(spec.target)}"
                    )
                if not isinstance(spec.arg, TensorArgument):
                    raise AssertionError(
                        f"expected TensorArgument for GRADIENT_TO_USER_INPUT, got {type(spec.arg)}"
                    )
                gradients_to_user_inputs[spec.arg.name] = spec.target

        if loss_output is None:
            return None

        return ExportBackwardSignature(
            loss_output=loss_output,
            gradients_to_parameters=gradients_to_parameters,
            gradients_to_user_inputs=gradients_to_user_inputs,
        )

    # Map from assertion dependency token index to assertion dep token output
    # name in output. The shape of output after aot_autograd will be like:
    # (updated_inputs, user_outputs, dep_token).
    @property
    def assertion_dep_token(self) -> Mapping[int, str] | None:
        return None

    @property
    def input_tokens(self) -> Collection[str]:
        input_tokens = []
        for s in self.input_specs:
            if s.kind == InputKind.TOKEN:
                if not isinstance(s.arg, TokenArgument):
                    raise AssertionError(
                        f"expected TokenArgument for TOKEN kind, got {type(s.arg)}"
                    )
                input_tokens.append(s.arg.name)
        return tuple(input_tokens)

    @property
    def output_tokens(self) -> Collection[str]:
        output_tokens = []
        for s in self.output_specs:
            if s.kind == OutputKind.TOKEN:
                if not isinstance(s.arg, TokenArgument):
                    raise AssertionError(
                        f"expected TokenArgument for TOKEN kind, got {type(s.arg)}"
                    )
                output_tokens.append(s.arg.name)
        return tuple(output_tokens)

    def __post_init__(self) -> None:
        assertion_dep_token = self.assertion_dep_token
        if assertion_dep_token is None:
            return
        if len(assertion_dep_token) != 1:
            raise AssertionError(
                f"expected exactly 1 assertion_dep_token, got {len(assertion_dep_token)}"
            )
        assertion_dep_token_index = next(iter(assertion_dep_token.keys()))
        expected_index = len(self.user_outputs) + len(self.buffers_to_mutate)
        if expected_index != assertion_dep_token_index:
            raise AssertionError(
                f"expected assertion_dep_token_index to be {expected_index}, got {assertion_dep_token_index}"
            )

    def replace_all_uses(self, old: str, new: str):
        """
        Replace all uses of the old name with new name in the signature.
        """
        if not isinstance(old, str):
            raise AssertionError(f"expected old to be str, got {type(old)}")
        if not isinstance(new, str):
            raise AssertionError(f"expected new to be str, got {type(new)}")
        arg_types = (
            TensorArgument,
            SymIntArgument,
            SymFloatArgument,
            SymBoolArgument,
            CustomObjArgument,
            TokenArgument,
        )
        for o in self.output_specs:
            if isinstance(o.arg, arg_types):
                if o.arg.name == old:
                    o.arg.name = new
        for i in self.input_specs:
            if isinstance(i.arg, arg_types):
                if i.arg.name == old:
                    i.arg.name = new

    def get_replace_hook(self, replace_inputs=False):
        def _(old, new, user):
            if user.op == "output":
                self.replace_all_uses(old.name, new)
            if replace_inputs and old.op == "placeholder":
                self.replace_all_uses(old.name, new)

        return _

    def __str__(self):
        input_specs = "\n".join(str(s) for s in self.input_specs)
        output_specs = "\n".join(str(s) for s in self.output_specs)
        return f"\n# inputs\n{input_specs}\n\n# outputs\n{output_specs}\n"


def _immutable_dict(items):
    """
    Creates a mapping where items cannot be added, deleted, or updated.
    NOTE: The immutability is shallow (like tuple is an immutable collection).
    """
    from types import MappingProxyType

    return MappingProxyType(dict(items))


def _make_argument_spec(node, token_names) -> ArgumentSpec:
    from torch import ScriptObject, SymBool, SymFloat, SymInt
    from torch._library.fake_class_registry import FakeScriptObject

    if isinstance(node, (int, bool, float, type(None), str)):
        # For const outputs we just directly return this
        return ConstantArgument(name="", value=node)

    if "val" not in node.meta:
        raise AssertionError(
            f"{node} is not a constant or a node with a 'val' metadata field"
        )
    val = node.meta["val"]
    if node.name in token_names:
        return TokenArgument(name=node.name)
    elif is_fake(val):
        return TensorArgument(name=node.name)
    elif isinstance(val, SymInt):
        return SymIntArgument(name=node.name)
    elif isinstance(val, SymFloat):
        return SymFloatArgument(name=node.name)
    elif isinstance(val, SymBool):
        return SymBoolArgument(name=node.name)
    elif isinstance(val, ScriptObject):
        return CustomObjArgument(name=node.name, class_fqn=val._type().qualified_name())  # type: ignore[attr-defined]
    elif isinstance(val, FakeScriptObject):
        return CustomObjArgument(
            name=node.name, class_fqn=val.script_class_name, fake_val=val
        )
    elif is_opaque_type(type(val)):
        return CustomObjArgument(
            name=node.name, class_fqn=get_opaque_type_name(type(val)), fake_val=val
        )
    elif isinstance(val, (int, bool, str, float, type(None))):
        return ConstantArgument(name=node.name, value=val)
    else:
        raise AssertionError(
            f"Encountered an unsupported object of type {type(val)} "
            f"while writing the metadata for exported program"
        )


def _convert_to_export_graph_signature(
    graph_signature: "GraphSignature",
    gm: "torch.fx.GraphModule",
    non_persistent_buffers: set[str],
) -> "ExportGraphSignature":
    from torch.utils import _pytree as pytree

    is_joint = graph_signature.backward_signature is not None

    # unpack objects
    user_inputs = set(graph_signature.user_inputs)
    inputs_to_parameters = graph_signature.inputs_to_parameters
    inputs_to_buffers = graph_signature.inputs_to_buffers
    user_outputs = set(graph_signature.user_outputs)
    buffer_mutations = graph_signature.buffers_to_mutate
    parameter_mutations = graph_signature.parameters_to_mutate
    user_input_mutations = graph_signature.user_inputs_to_mutate
    grad_params = (
        graph_signature.backward_signature.gradients_to_parameter  # type: ignore[union-attr]
        if is_joint
        else {}
    )
    grad_user_inputs = (
        graph_signature.backward_signature.gradients_to_user_inputs  # type: ignore[union-attr]
        if is_joint
        else {}
    )
    loss_output = (
        graph_signature.backward_signature.loss_output  # type: ignore[union-attr]
        if is_joint
        else None
    )
    input_tokens = graph_signature.input_tokens
    output_tokens = graph_signature.output_tokens

    inputs = [
        _make_argument_spec(node, input_tokens)
        for node in gm.graph.nodes
        if node.op == "placeholder"
    ]
    outputs = [
        _make_argument_spec(node, output_tokens)
        for node in pytree.tree_leaves(next(iter(reversed(gm.graph.nodes))).args)
    ]

    def to_input_spec(inp: ArgumentSpec) -> InputSpec:
        if isinstance(inp, TokenArgument):
            return InputSpec(kind=InputKind.TOKEN, arg=inp, target=None)

        if not isinstance(inp, TensorArgument):
            return InputSpec(kind=InputKind.USER_INPUT, arg=inp, target=None)
        name = inp.name
        if name in user_inputs:
            return InputSpec(kind=InputKind.USER_INPUT, arg=inp, target=None)
        elif name in inputs_to_parameters:
            return InputSpec(
                kind=InputKind.PARAMETER,
                arg=inp,
                target=inputs_to_parameters[name],  # type: ignore[index]
            )
        elif name in inputs_to_buffers:
            return InputSpec(
                kind=InputKind.BUFFER,
                arg=inp,
                target=inputs_to_buffers[name],  # type: ignore[index]
                persistent=(inputs_to_buffers[name] not in non_persistent_buffers),  # type: ignore[index]
            )
        else:
            raise AssertionError(f"Unknown tensor input kind: {name}")

    def to_output_spec(idx: int, o: ArgumentSpec) -> OutputSpec:
        if isinstance(o, TokenArgument):
            return OutputSpec(kind=OutputKind.TOKEN, arg=o, target=None)

        if not isinstance(o, TensorArgument):
            return OutputSpec(kind=OutputKind.USER_OUTPUT, arg=o, target=None)
        name = o.name
        if idx < len(buffer_mutations) + len(parameter_mutations) + len(
            user_input_mutations
        ) + len(output_tokens):
            if name in buffer_mutations:
                return OutputSpec(
                    kind=OutputKind.BUFFER_MUTATION,
                    arg=o,
                    target=buffer_mutations[name],  # type: ignore[index]
                )
            elif name in parameter_mutations:
                return OutputSpec(
                    kind=OutputKind.PARAMETER_MUTATION,
                    arg=o,
                    target=parameter_mutations[name],  # type: ignore[index]
                )
            elif name in user_input_mutations:
                return OutputSpec(
                    kind=OutputKind.USER_INPUT_MUTATION,
                    arg=o,
                    target=user_input_mutations[name],  # type: ignore[index]
                )
            else:
                raise AssertionError(f"Unknown tensor mutation kind: {name}")
        else:
            if name in user_outputs:
                return OutputSpec(kind=OutputKind.USER_OUTPUT, arg=o, target=None)

            elif name in grad_params:
                return OutputSpec(
                    kind=OutputKind.GRADIENT_TO_PARAMETER,
                    arg=o,
                    target=grad_params[name],
                )
            elif name in grad_user_inputs:
                return OutputSpec(
                    kind=OutputKind.GRADIENT_TO_USER_INPUT,
                    arg=o,
                    target=grad_user_inputs[name],
                )
            elif name == loss_output:
                return OutputSpec(kind=OutputKind.LOSS_OUTPUT, arg=o, target=None)

            else:
                raise AssertionError(f"Unknown tensor output kind: {name}")

    input_specs = [to_input_spec(inp) for inp in inputs]
    output_specs = [to_output_spec(idx, o) for idx, o in enumerate(outputs)]
    return ExportGraphSignature(input_specs=input_specs, output_specs=output_specs)
