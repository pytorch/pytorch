# mypy: allow-untyped-defs
import dataclasses
from enum import auto, Enum
from typing import Collection, Dict, List, Mapping, Optional, Set, TYPE_CHECKING, Union

from torch._library.fake_class_registry import FakeScriptObject


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
class CustomObjArgument:
    name: str
    class_fqn: str
    fake_val: Optional[FakeScriptObject] = None


@dataclasses.dataclass
class ConstantArgument:
    name: str
    value: Union[int, float, bool, str, None]


ArgumentSpec = Union[
    TensorArgument,
    SymIntArgument,
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
    target: Optional[str]
    persistent: Optional[bool] = None

    def __post_init__(self):
        if self.kind == InputKind.BUFFER:
            assert (
                self.persistent is not None
            ), "Failed to specify persistent flag on BUFFER."
        assert isinstance(
            self.arg,
            (
                TensorArgument,
                SymIntArgument,
                ConstantArgument,
                CustomObjArgument,
                TokenArgument,
            ),
        ), f"got {type(self.arg)}"


class OutputKind(Enum):
    USER_OUTPUT = auto()
    LOSS_OUTPUT = auto()
    BUFFER_MUTATION = auto()
    GRADIENT_TO_PARAMETER = auto()
    GRADIENT_TO_USER_INPUT = auto()
    USER_INPUT_MUTATION = auto()
    TOKEN = auto()


@dataclasses.dataclass
class OutputSpec:
    kind: OutputKind
    arg: ArgumentSpec
    target: Optional[str]

    def __post_init__(self):
        assert isinstance(
            self.arg,
            (
                TensorArgument,
                SymIntArgument,
                ConstantArgument,
                TokenArgument,
                CustomObjArgument,
            ),
        ), self.arg


@dataclasses.dataclass
class ExportBackwardSignature:
    gradients_to_parameters: Dict[str, str]
    gradients_to_user_inputs: Dict[str, str]
    loss_output: str


@dataclasses.dataclass
class ExportGraphSignature:
    """
    :class:`ExportGraphSignature` models the input/output signature of Export Graph,
    which is a fx.Graph with stronger invariants gurantees.

    Export Graph is functional and does not access "states" like parameters
    or buffers within the graph via ``getattr`` nodes. Instead, :func:`export`
    gurantees that parameters, buffers, and constant tensors are lifted out of
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
                self.register_buffer('my_buffer1', torch.tensor(3.0))
                self.register_buffer('my_buffer2', torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (x1 + self.my_parameter) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0) # In-place addition

                return output

    Resulting Graph would be::

        graph():
            %arg0_1 := placeholder[target=arg0_1]
            %arg1_1 := placeholder[target=arg1_1]
            %arg2_1 := placeholder[target=arg2_1]
            %arg3_1 := placeholder[target=arg3_1]
            %arg4_1 := placeholder[target=arg4_1]
            %add_tensor := call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, %arg0_1), kwargs = {})
            %mul_tensor := call_function[target=torch.ops.aten.mul.Tensor](args = (%add_tensor, %arg1_1), kwargs = {})
            %mul_tensor_1 := call_function[target=torch.ops.aten.mul.Tensor](args = (%arg4_1, %arg2_1), kwargs = {})
            %add_tensor_1 := call_function[target=torch.ops.aten.add.Tensor](args = (%mul_tensor, %mul_tensor_1), kwargs = {})
            %add_tensor_2 := call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, 1.0), kwargs = {})
            return (add_tensor_2, add_tensor_1)

    Resulting ExportGraphSignature would be::

        ExportGraphSignature(
            input_specs=[
                InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='arg0_1'), target='my_parameter'),
                InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='arg1_1'), target='my_buffer1'),
                InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='arg2_1'), target='my_buffer2'),
                InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='arg3_1'), target=None),
                InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='arg4_1'), target=None)
            ],
            output_specs=[
                OutputSpec(kind=<OutputKind.BUFFER_MUTATION: 3>, arg=TensorArgument(name='add_2'), target='my_buffer2'),
                OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='add_1'), target=None)
            ]
        )
    """

    input_specs: List[InputSpec]
    output_specs: List[OutputSpec]

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
    def user_inputs(self) -> Collection[Union[int, float, bool, None, str]]:
        user_inputs: List[Union[int, float, bool, None, str]] = []
        for s in self.input_specs:
            if s.kind != InputKind.USER_INPUT:
                continue

            if isinstance(s.arg, (TensorArgument, SymIntArgument, CustomObjArgument)):
                user_inputs.append(s.arg.name)
            elif isinstance(s.arg, ConstantArgument):
                user_inputs.append(s.arg.value)
            else:
                raise RuntimeError(f"{s.arg} is not a valid user inputs")
        return tuple(user_inputs)

    # Graph node names of pytree-flattened outputs of original program
    @property
    def user_outputs(self) -> Collection[Union[int, float, bool, None, str]]:
        user_outputs: List[Union[int, float, bool, None, str]] = []
        for s in self.output_specs:
            if s.kind != OutputKind.USER_OUTPUT:
                continue

            if isinstance(s.arg, (TensorArgument, SymIntArgument)):
                user_outputs.append(s.arg.name)
            elif isinstance(s.arg, ConstantArgument):
                user_outputs.append(s.arg.value)
            elif isinstance(s.arg, CustomObjArgument):
                user_outputs.append(s.arg.name)
            else:
                raise RuntimeError(f"{s.arg} is not a valid user output")
        return tuple(user_outputs)

    # A dictionary mapping graph input node names to parameters. If a graph input
    # name is found in this dictionary, it is guranteed to be a lifted parameter.
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
    # name is found in this dictionary, it is guranteed to be a lifted buffer.
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
    def backward_signature(self) -> Optional[ExportBackwardSignature]:
        loss_output = None
        gradients_to_parameters: Dict[str, str] = {}
        gradients_to_user_inputs: Dict[str, str] = {}
        for spec in self.output_specs:
            if spec.kind == OutputKind.LOSS_OUTPUT:
                assert loss_output is None
                assert isinstance(spec.arg, TensorArgument)
                loss_output = spec.arg.name
            elif spec.kind == OutputKind.GRADIENT_TO_PARAMETER:
                assert isinstance(spec.target, str)
                assert isinstance(spec.arg, TensorArgument)
                gradients_to_parameters[spec.arg.name] = spec.target
            elif spec.kind == OutputKind.GRADIENT_TO_USER_INPUT:
                assert isinstance(spec.target, str)
                assert isinstance(spec.arg, TensorArgument)
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
    def assertion_dep_token(self) -> Optional[Mapping[int, str]]:
        return None

    @property
    def input_tokens(self) -> Collection[str]:
        input_tokens = []
        for s in self.input_specs:
            if s.kind == InputKind.TOKEN:
                assert isinstance(s.arg, TokenArgument)
                input_tokens.append(s.arg.name)
        return tuple(input_tokens)

    @property
    def output_tokens(self) -> Collection[str]:
        output_tokens = []
        for s in self.output_specs:
            if s.kind == OutputKind.TOKEN:
                assert isinstance(s.arg, TokenArgument)
                output_tokens.append(s.arg.name)
        return tuple(output_tokens)

    def __post_init__(self) -> None:
        assertion_dep_token = self.assertion_dep_token
        if assertion_dep_token is None:
            return
        assert len(assertion_dep_token) == 1
        assertion_dep_token_index = next(iter(assertion_dep_token.keys()))
        assert (
            len(self.user_outputs) + len(self.buffers_to_mutate)
            == assertion_dep_token_index
        )

    def replace_all_uses(self, old: str, new: str):
        """
        Replace all uses of the old name with new name in the signature.
        """
        assert isinstance(old, str)
        assert isinstance(new, str)
        arg_types = (TensorArgument, SymIntArgument, CustomObjArgument, TokenArgument)
        for o in self.output_specs:
            if isinstance(o.arg, arg_types):
                if o.arg.name == old:
                    o.arg.name = new
        for i in self.input_specs:
            if isinstance(i.arg, arg_types):
                if i.arg.name == old:
                    i.arg.name = new

    def get_replace_hook(self):
        def _(old, new, user):
            if user.op in ("output", "input"):
                self.replace_all_uses(old.name, new)

        return _


def _immutable_dict(items):
    """
    Creates a mapping where items cannot be added, deleted, or updated.
    NOTE: The immutability is shallow (like tuple is an immutable collection).
    """
    from types import MappingProxyType

    return MappingProxyType(dict(items))


def _make_argument_spec(node, token_names) -> ArgumentSpec:
    from torch import ScriptObject, SymInt
    from torch._library.fake_class_registry import FakeScriptObject
    from torch._subclasses.fake_tensor import FakeTensor

    if isinstance(node, (int, bool, float, type(None), str)):
        # For const outputs we just directly return this
        return ConstantArgument(name="", value=node)

    assert (
        "val" in node.meta
    ), f"{node} is not a constant or a node with a 'val' metadata field"
    val = node.meta["val"]
    if node.name in token_names:
        return TokenArgument(name=node.name)
    elif isinstance(val, FakeTensor):
        return TensorArgument(name=node.name)
    elif isinstance(val, SymInt):
        return SymIntArgument(name=node.name)
    elif isinstance(val, ScriptObject):
        return CustomObjArgument(name=node.name, class_fqn=val._type().qualified_name())  # type: ignore[attr-defined]
    elif isinstance(val, FakeScriptObject):
        return CustomObjArgument(
            name=node.name, class_fqn=val.script_class_name, fake_val=val
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
    non_persistent_buffers: Set[str],
) -> "ExportGraphSignature":
    from torch.utils import _pytree as pytree

    is_joint = graph_signature.backward_signature is not None

    # unpack objects
    user_inputs = set(graph_signature.user_inputs)
    inputs_to_parameters = graph_signature.inputs_to_parameters
    inputs_to_buffers = graph_signature.inputs_to_buffers
    user_outputs = set(graph_signature.user_outputs)
    buffer_mutations = graph_signature.buffers_to_mutate
    user_input_mutations = graph_signature.user_inputs_to_mutate
    grad_params = graph_signature.backward_signature.gradients_to_parameter if is_joint else {}  # type: ignore[union-attr]
    grad_user_inputs = graph_signature.backward_signature.gradients_to_user_inputs if is_joint else {}  # type: ignore[union-attr]
    loss_output = graph_signature.backward_signature.loss_output if is_joint else None  # type: ignore[union-attr]
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
        if idx < len(buffer_mutations) + len(user_input_mutations) + len(output_tokens):
            if name in buffer_mutations:
                return OutputSpec(
                    kind=OutputKind.BUFFER_MUTATION,
                    arg=o,
                    target=buffer_mutations[name],  # type: ignore[index]
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
