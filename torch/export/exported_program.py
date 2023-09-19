import copy
import dataclasses
from enum import auto, Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import sympy

import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility

from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager


__all__ = [
    "ArgumentKind",
    "ArgumentSpec",
    "ExportBackwardSignature",
    "ExportedProgram",
    "ExportGraphSignature",
    "ModuleCallEntry",
    "ModuleCallSignature",
]


PassType = Callable[[torch.fx.GraphModule], Optional[PassResult]]


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
    gurantees that parameters and buffers are lifted out of the graph as inputs.
    Similarly, any mutations to buffers are not included in the graph either,
    instead the updated values of mutated buffers are modeled as additional outputs
    of Export Graph.

    The ordering of all inputs and outputs are::

        Inputs = [*parameters_buffers, *flattened_user_inputs]
        Outputs = [*mutated_inputs, *flattened_user_outputs]

    e.g. If following module is exported::

        class CustomModule(nn.Module):
            def __init__(self):
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
            # Indicates that there is one parameter named `my_parameter`
            parameters=['L__self___my_parameter'],

            # Indicates that there are two buffers, `my_buffer1` and `my_buffer2`
            buffers=['L__self___my_buffer1', 'L__self___my_buffer2'],

            # Indicates that the nodes `arg3_1` and `arg4_1` in produced graph map to
            # original user inputs, ie. x1 and x2
            user_inputs=['arg3_1', 'arg4_1'],

            # Indicates that the node `add_tensor_1` maps to output of original program
            user_outputs=['add_tensor_1'],

            # Indicates that there is one parameter (self.my_parameter) captured,
            # its name is now mangled to be `L__self___my_parameter`, which is now
            # represented by node `arg0_1` in the graph.
            inputs_to_parameters={'arg0_1': 'L__self___my_parameter'},

            # Indicates that there are two buffers (self.my_buffer1, self.my_buffer2) captured,
            # their name are now mangled to be `L__self___my_my_buffer1` and `L__self___my_buffer2`.
            # They are now represented by nodes `arg1_1` and `arg2_1` in the graph.
            inputs_to_buffers={'arg1_1': 'L__self___my_buffer1', 'arg2_1': 'L__self___my_buffer2'},

            # Indicates that one buffer named `L__self___my_buffer2` is mutated during execution,
            # its new value is output from the graph represented by the node named `add_tensor_2`
            buffers_to_mutate={'add_tensor_2': 'L__self___my_buffer2'},

            # Backward graph not captured
            backward_signature=None,

            # Work in progress feature, please ignore now.
            assertion_dep_token=None
        )
    """

    # A list of parameters uniquely identified by mangled fully qualified name
    parameters: List[str]

    # A list of buffers uniquely identified by mangled fully qualified name
    buffers: List[str]

    # Graph node names of pytree-flattened inputs of original program
    user_inputs: List[str]

    # Graph node names of pytree-flattened outputs of original program
    user_outputs: List[str]

    # A dictionary mapping graph input node names to parameters. If a graph input
    # name is found in this dictionary, it is guranteed to be a lifted parameter.
    inputs_to_parameters: Dict[str, str]

    # A dictionary mapping graph input node names to buffers. If a graph input
    # name is found in this dictionary, it is guranteed to be a lifted buffer.
    inputs_to_buffers: Dict[str, str]

    # A dictionary mapping graph output node names to buffers that are mutated in the
    # original program. Buffers that are not mutated will not be found in this dictionary.
    buffers_to_mutate: Dict[str, str]

    backward_signature: Optional[ExportBackwardSignature]

    # Map from assertion dependency token index to assertion dep token output
    # name in output. The shape of output after aot_autograd will be like:
    # (updated_inputs, user_outputs, dep_token).
    assertion_dep_token: Optional[Dict[int, str]] = None

    def __post_init__(self) -> None:
        assertion_dep_token = self.assertion_dep_token
        if assertion_dep_token is None:
            return
        assert len(assertion_dep_token) == 1
        assertion_dep_token_index = list(assertion_dep_token.keys())[0]
        assert (
            len(self.user_outputs) + len(self.buffers_to_mutate)
            == assertion_dep_token_index
        )


class ArgumentKind(Enum):
    Tensor = auto()
    SymInt = auto()
    Constant = auto()


@dataclasses.dataclass
class ArgumentSpec:
    kind: ArgumentKind
    value: Any

    def __post_init__(self):
        if self.kind in (ArgumentKind.Tensor, ArgumentKind.SymInt):
            assert isinstance(self.value, str)


@dataclasses.dataclass
class ModuleCallSignature:
    inputs: List[ArgumentSpec]
    outputs: List[ArgumentSpec]
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec


@dataclasses.dataclass
class ModuleCallEntry:
    fqn: str
    signature: Optional[ModuleCallSignature] = None


class ExportedProgram:
    """
    Package of a program from :func:`export`. It contains
    an :class:`torch.fx.Graph` that represents Tensor computation, a state_dict containing
    tensor values of all lifted parameters and buffers, and various metadata.

    You can call an ExportedProgram like the original callable traced by
    :func:`export` with the same calling convention.

    To perform transformations on the graph, use ``.module`` property to access
    an :class:`torch.fx.GraphModule`. You can then use
    `FX transformation <https://pytorch.org/docs/stable/fx.html#writing-transformations>`_
    to rewrite the graph. Afterwards, you can simply use :func:`export`
    again to construct a correct ExportedProgram.
    """

    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: torch.fx.Graph,
        graph_signature: ExportGraphSignature,
        call_spec: Any,
        state_dict: Dict[str, Union[torch.Tensor, torch.nn.Parameter]],
        range_constraints: Dict[sympy.Symbol, Any],
        equality_constraints: List[Tuple[Any, Any]],
        module_call_graph: List[ModuleCallEntry],
        example_inputs: Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]] = None,
        dialect: Optional[str] = None,
    ):
        from torch._export.exported_program import (
            _create_graph_module_for_export,
            CallSpec,
        )
        from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
            InputDim,
            RangeConstraint,
        )

        # Remove codegen related things from the graph. It should just be a flat graph.
        graph._codegen = torch.fx.graph.CodeGen()
        self._graph_module = _create_graph_module_for_export(root, graph)
        if isinstance(root, torch.fx.GraphModule):
            self._graph_module.meta.update(root.meta)

        self._graph_signature: ExportGraphSignature = graph_signature
        self._call_spec: CallSpec = call_spec
        self._state_dict: Dict[str, Any] = state_dict
        self._range_constraints: Dict[sympy.Symbol, RangeConstraint] = range_constraints
        self._equality_constraints: List[
            Tuple[InputDim, InputDim]
        ] = equality_constraints
        self._module_call_graph: List[ModuleCallEntry] = module_call_graph
        self._example_inputs = example_inputs
        self._dialect = dialect or "ATEN"

    @property
    @compatibility(is_backward_compatible=False)
    def graph_module(self):
        return self._graph_module

    @property
    @compatibility(is_backward_compatible=False)
    def graph(self):
        return self.graph_module.graph

    @property
    @compatibility(is_backward_compatible=False)
    def graph_signature(self):
        return self._graph_signature

    @property
    @compatibility(is_backward_compatible=False)
    def state_dict(self):
        return self._state_dict

    @compatibility(is_backward_compatible=False)
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """
        Returns an iterator over original module's parameters.
        """
        for _, param in self.named_parameters():
            yield param

    @compatibility(is_backward_compatible=False)
    def named_parameters(self) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """
        Returns an iterator over original module parameters, yielding
        both the name of the parameter as well as the parameter itself.
        """
        for param_name in self.graph_signature.parameters:
            yield param_name, self.state_dict[param_name]

    @compatibility(is_backward_compatible=False)
    def buffers(self) -> Iterator[torch.Tensor]:
        """
        Returns an iterator over original module buffers.
        """
        for _, buf in self.named_buffers():
            yield buf

    @compatibility(is_backward_compatible=False)
    def named_buffers(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Returns an iterator over original module buffers, yielding
        both the name of the buffer as well as the buffer itself.
        """
        for buffer_name in self.graph_signature.buffers:
            yield buffer_name, self.state_dict[buffer_name]

    @property
    @compatibility(is_backward_compatible=False)
    def call_spec(self):
        return self._call_spec

    @property
    @compatibility(is_backward_compatible=False)
    def range_constraints(self):
        return self._range_constraints

    @property
    @compatibility(is_backward_compatible=False)
    def equality_constraints(self):
        return self._equality_constraints

    @property
    @compatibility(is_backward_compatible=False)
    def module_call_graph(self):
        return self._module_call_graph

    @property
    @compatibility(is_backward_compatible=False)
    def example_inputs(self):
        return self._example_inputs

    @property
    def dialect(self):
        return self._dialect

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch._export.error as error
        from torch._export import combine_args_kwargs

        if self.call_spec.in_spec is not None:
            try:
                user_args = combine_args_kwargs(args, kwargs)
                args = fx_pytree.tree_flatten_spec(user_args, self.call_spec.in_spec)  # type: ignore[assignment]
            except Exception:
                _, received_spec = pytree.tree_flatten(user_args)
                raise error.InternalError(
                    "Trying to flatten user inputs with exported input tree spec: \n"
                    f"{self.call_spec.in_spec}\n"
                    "but actually got inputs with tree spec of: \n"
                    f"{received_spec}"
                )

        ordered_params = tuple(
            self.state_dict[name] for name in self.graph_signature.parameters
        )
        ordered_buffers = tuple(
            self.state_dict[name] for name in self.graph_signature.buffers
        )
        self._check_input_constraints(*ordered_params, *ordered_buffers, *args)

        with torch.no_grad():
            # NOTE: calling convention is first params, then buffers, then args as user supplied them.
            # See: torch/_functorch/aot_autograd.py#L1034
            res = torch.fx.Interpreter(self.graph_module).run(
                *ordered_params, *ordered_buffers, *args, enable_io_processing=False
            )

        if self.call_spec.out_spec is not None:
            mutation = self.graph_signature.buffers_to_mutate
            num_mutated = len(mutation)
            mutated_buffers = res[:num_mutated]

            # Exclude dependency token from final result.
            assertion_dep_token = self.graph_signature.assertion_dep_token
            if assertion_dep_token is not None:
                assertion_dep_token_index = list(assertion_dep_token.keys())[0]
                res = res[:assertion_dep_token_index]

            res = res[num_mutated:]
            try:
                res = pytree.tree_unflatten(res, self.call_spec.out_spec)
            except Exception:
                _, received_spec = pytree.tree_flatten(res)
                raise error.InternalError(
                    "Trying to flatten user outputs with exported output tree spec: \n"
                    f"{self.call_spec.out_spec}\n"
                    "but actually got outputs with tree spec of: \n"
                    f"{received_spec}"
                )
            finally:
                ix = 0
                for buffer in self.graph_signature.buffers_to_mutate.values():
                    self.state_dict[buffer] = mutated_buffers[ix]
                    ix += 1
        return res

    def __str__(self) -> str:
        graph_module = self.graph_module.print_readable(print_output=False).replace(
            "\n", "\n    "
        )
        string = (
            "ExportedProgram:\n"
            f"    {graph_module}\n"
            f"Graph signature: {self.graph_signature}\n"
            f"Range constraints: {self.range_constraints}\n"
            f"Equality constraints: {self.equality_constraints}\n"
        )
        return string

    def module(self) -> torch.nn.Module:
        """
        Returns a self contained GraphModule with all the parameters/buffers inlined.
        """
        from torch._export.exported_program import unlift_exported_program_lifted_states

        return unlift_exported_program_lifted_states(self)

    def _transform(self, *passes: PassType) -> "ExportedProgram":
        from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
            RangeConstraint,
        )

        pm = PassManager(list(passes))
        res = pm(self.graph_module)
        transformed_gm = res.graph_module if res is not None else self.graph_module
        assert transformed_gm is not None

        def _get_updated_range_constraints(
            gm: torch.fx.GraphModule,
        ) -> Dict[sympy.Symbol, RangeConstraint]:
            def get_shape_env(gm):
                vals = [
                    node.meta["val"]
                    for node in gm.graph.nodes
                    if node.meta.get("val", None) is not None
                ]
                from torch._guards import detect_fake_mode

                fake_mode = detect_fake_mode(vals)
                if fake_mode is not None:
                    return fake_mode.shape_env
                for v in vals:
                    if isinstance(v, torch.SymInt):
                        return v.node.shape_env

            shape_env = get_shape_env(gm)
            if shape_env is None:
                return {}
            range_constraints = {
                k: RangeConstraint(v.lower, v.upper)
                for k, v in shape_env.var_to_range.items()
            }
            return range_constraints

        def _get_updated_graph_signature(
            old_signature: ExportGraphSignature,
            new_gm: torch.fx.GraphModule,
        ) -> ExportGraphSignature:
            """
            Update the graph signature's user_input/user_outputs.
            """
            new_graph_inputs = [
                node.name for node in new_gm.graph.nodes if node.op == "placeholder"
            ]
            num_inputs = (
                len(old_signature.parameters)
                + len(old_signature.buffers)
                + len(old_signature.user_inputs)
            )

            assert len(new_graph_inputs) == num_inputs, (
                f"Number of input nodes changed from {len(new_graph_inputs)} "
                f"to {num_inputs} after transformation. This transformation "
                "is currently not supported."
            )
            new_parameter_inputs = new_graph_inputs[: len(old_signature.parameters)]
            num_param_buffers = len(old_signature.buffers) + len(
                old_signature.parameters
            )
            new_buffer_inputs = new_graph_inputs[
                len(old_signature.parameters) : num_param_buffers
            ]
            new_user_inputs = new_graph_inputs[num_param_buffers:]

            output_node = list(new_gm.graph.nodes)[-1]
            assert output_node.op == "output"
            new_graph_outputs = [arg.name for arg in output_node.args[0]]

            assert len(new_graph_outputs) == len(old_signature.buffers_to_mutate) + len(
                old_signature.user_outputs
            ), (
                f"Number of output nodes changed from {len(new_graph_outputs)} "
                f"to {len(old_signature.buffers_to_mutate) + len(old_signature.user_outputs)} "
                "after transformation. This transformation is currently not supported."
            )
            new_user_outputs = new_graph_outputs[len(old_signature.buffers_to_mutate) :]

            new_signature = ExportGraphSignature(
                copy.deepcopy(old_signature.parameters),
                copy.deepcopy(old_signature.buffers),
                new_user_inputs,
                new_user_outputs,
                copy.deepcopy(old_signature.inputs_to_parameters),
                copy.deepcopy(old_signature.inputs_to_buffers),
                copy.deepcopy(old_signature.buffers_to_mutate),
                copy.deepcopy(old_signature.backward_signature),
                copy.deepcopy(old_signature.assertion_dep_token),
            )
            return new_signature

        transformed_ep = ExportedProgram(
            transformed_gm,
            transformed_gm.graph,
            _get_updated_graph_signature(self.graph_signature, transformed_gm),
            copy.deepcopy(self.call_spec),
            self.state_dict,
            _get_updated_range_constraints(transformed_gm),
            copy.deepcopy(self.equality_constraints),
            copy.deepcopy(self._module_call_graph),
            self.example_inputs,
            self.dialect,
        )
        transformed_ep.graph_module.meta.update(self.graph_module.meta)
        transformed_ep.graph_module.meta.update(res.graph_module.meta)
        return transformed_ep

    def _check_input_constraints(self, *args):
        from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
            _AddRuntimeAssertionsForConstraintsPass,
        )

        # TODO(zhxchen17) Don't generate a runtime graph on the fly.
        _assertion_graph = torch.fx.GraphModule({}, torch.fx.Graph())
        for p in self.graph.nodes:
            if p.op != "placeholder":
                continue
            new_p = _assertion_graph.graph.placeholder(p.name)
            new_p.meta = p.meta
        _assertion_graph.graph.output(())
        _assertion_graph_res = _AddRuntimeAssertionsForConstraintsPass(
            self.range_constraints,
            self.equality_constraints,
        )(_assertion_graph)
        assert _assertion_graph_res is not None
        _assertion_graph = _assertion_graph_res.graph_module
        _assertion_graph(*args)

    def _validate(self):
        # TODO(zhxchen17) check for get_attr
        # TODO(zhxchen17) check for funcitonal ops
        for gm in self.graph_module.modules():
            if not isinstance(gm, torch.fx.GraphModule):
                continue
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    assert node.target != torch.ops.higher_order._export_tracepoint
