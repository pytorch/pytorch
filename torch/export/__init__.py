import copy
import dataclasses
import io
import pathlib
import typing
from enum import auto, Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sympy

import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility

from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint

from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager


__all__ = [
    "ArgumentKind",
    "ArgumentSpec",
    "Constraint",
    "ExportBackwardSignature",
    "ExportGraphSignature",
    "ExportedProgram",
    "ModuleCallEntry",
    "ModuleCallSignature",
    "constrain_as_size",
    "constrain_as_value",
    "dynamic_dim",
    "export",
    "load",
    "save",
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
    ExportGraphSignature models the input/output signature of Export Graph,
    which is a fx.Graph with stronger invariants gurantees.

    Export Graph is functional and does not access "states" like parameters
    or buffers within the graph via `getattr` nodes. Instead, torch.export()
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
    Package of a program from :func:`torch.export.export()`. It contains
    an fx.Graph that represents Tensor computation, a state_dict containing
    tensor values of all lifted parameters and buffers, and various metadata.

    You can call an ExportedProgram like the original callable traced by
    :func:`torch.export.export()` with the same calling convention.

    To perform transformations on the graph, use `.module` property to access
    an :class:`torch.fx.GraphModule`. You can then use
    `FX transformation <https://pytorch.org/docs/stable/fx.html#writing-transformations>`_
    to rewrite the graph. Afterwards, you can simply use :func:`torch.export.export()`
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
            f"Graph Signature: {self.graph_signature}\n"
            f"Symbol to range: {self.range_constraints}\n"
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

        def get_output_node_names(gm):
            output_node = list(gm.graph.nodes)[-1]
            assert output_node.op == "output"

            return [str(arg) for arg in output_node.args[0]]

        def get_input_node_names(gm):
            return [node.name for node in gm.graph.nodes if node.op == "placeholder"]

        def _generate_new_graph_signature(old_ep, new_gm):
            """
            Update graph_signature according to graph after transformation.
            Transformations can lead to node name changes, which are used in
            graph_signature to identify inputs and outputs. Therefore, after each
            transformation, we need to update the graph_signature according to
            new node names.

            WARNING: This implementation makes a few assumptions
                - The transformation doesn't change number of inputs/outputs
                - Each input/output still has the same meaning.
                    - For inputs, that means that the inputs in transformed
                        graph map to the same lifted parameter/buffer or user
                        input as the input of the same position in the graph
                        before transformation.
                    - Similarly for outputs, each output should correspond to the
                        same mutated buffer or user output as the output value of
                        the same position  in the graph before transformation.

            It is difficult to programatically validate these assumptions, but they
            should hold true most of the time as inputs/outputs of the graph rarely
            need to be changed.
            """
            old_signature = old_ep.graph_signature
            old_gm = old_ep.graph_module

            old_graph_input_node_names = get_input_node_names(old_gm)
            new_graph_input_node_names = get_input_node_names(new_gm)
            assert len(old_graph_input_node_names) == len(
                new_graph_input_node_names
            ), f"""
                Number of input nodes changed from {len(old_graph_input_node_names)}
                to {len(new_graph_input_node_names)} after transformation. This
                transformation is currently not supported.
                """

            old_graph_output_node_names = get_output_node_names(old_gm)
            new_graph_output_node_names = get_output_node_names(new_gm)
            assert len(old_graph_output_node_names) == len(
                new_graph_output_node_names
            ), f"""
                Number of output values changed from {len(old_graph_output_node_names)}
                to {len(new_graph_output_node_names)} after transformation. This
                transformation is currently not supported.
                """

            node_names_mapping = dict(
                zip(
                    old_graph_input_node_names + old_graph_output_node_names,
                    new_graph_input_node_names + new_graph_output_node_names,
                )
            )

            new_signature = copy.deepcopy(old_signature)
            new_signature.user_inputs = [
                node_names_mapping[old_user_input]
                for old_user_input in old_signature.user_inputs
            ]
            new_signature.user_outputs = [
                node_names_mapping[old_user_output]
                for old_user_output in old_signature.user_outputs
            ]
            new_signature.inputs_to_parameters = {
                node_names_mapping[old_input_name]: old_signature.inputs_to_parameters[
                    old_input_name
                ]
                for old_input_name in old_signature.inputs_to_parameters.keys()
            }
            new_signature.inputs_to_buffers = {
                node_names_mapping[old_input_name]: old_signature.inputs_to_buffers[
                    old_input_name
                ]
                for old_input_name in old_signature.inputs_to_buffers.keys()
            }
            new_signature.buffers_to_mutate = {
                node_names_mapping[old_output_name]: old_signature.buffers_to_mutate[
                    old_output_name
                ]
                for old_output_name in old_signature.buffers_to_mutate.keys()
            }
            return new_signature

        new_graph_signature = _generate_new_graph_signature(self, transformed_gm)

        transformed_ep = ExportedProgram(
            transformed_gm,
            transformed_gm.graph,
            new_graph_signature,
            copy.deepcopy(self.call_spec),
            self.state_dict,
            _get_updated_range_constraints(transformed_gm),
            copy.deepcopy(self.equality_constraints),
            copy.deepcopy(self._module_call_graph),
            self.example_inputs,
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


@dataclasses.dataclass
class _ConstraintTarget:
    """
    This represents input tensor dimensions.  Don't create this
    class directly; instead, use :func:`torch.export.dynamic_dim`.
    """

    w_tensor: Any  # weakref to torch.Tensor
    # TODO: We don't need t_id; we can get it off of w_tensor
    t_id: int
    dim: int


class _ConstraintFactory(type):
    """
    Metaclass that ensures a private constructor for Constraint
    """

    def __call__(cls, *args, **kwargs):
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} has no public constructor. "
            f"Please use torch.export.dynamic_dim() to create one"
        )

    def _create(cls, w_tensor, t_id, dim, constraint_range, shared=None):
        return super().__call__(w_tensor, t_id, dim, constraint_range, shared)


def _create_constraint(w_tensor, t_id, dim, constraint_range, shared=None):
    return Constraint._create(w_tensor, t_id, dim, constraint_range, shared)


@dataclasses.dataclass
class Constraint(_ConstraintTarget, metaclass=_ConstraintFactory):
    """

    .. warning::
        Do not construct `Constraint` directly, use :func:`torch.export.dynamic_dim` instead.

    This represents constraints on input tensor dimensions, e.g., requiring
    them to be fully polymorphic or within some range.

    """

    # NOTE(avik): In the future, this could be Union[StrictMinMaxConstraint, <other kinds>]
    constraint_range: StrictMinMaxConstraint
    # Represent that `constraint_range` is shared with another _ConstraintTarget, which
    # typically arises because of a specified equality with another dynamic dimension.
    shared: Optional[_ConstraintTarget] = None

    def _clone_with_range(self, lower=2, upper=sympy.oo):
        from torch.utils._sympy.value_ranges import ValueRanges

        constraint_range = StrictMinMaxConstraint(
            vr=self.constraint_range.vr & ValueRanges(lower=lower, upper=upper),
            warn_only=False,
        )
        return _create_constraint(
            self.w_tensor, self.t_id, self.dim, constraint_range, self.shared
        )

    def __ge__(self, lower):
        return self._clone_with_range(lower=lower)

    def __gt__(self, lower):
        return self._clone_with_range(lower=lower + 1)

    def __le__(self, upper):
        return self._clone_with_range(upper=upper)

    def __lt__(self, upper):
        return self._clone_with_range(upper=upper - 1)

    def __bool__(self):
        # NOTE(avik): We do not support compound expressions like a <= x <= b.
        # This is because Python implicitly desugars them into bool(a <= x) and bool(x <= b),
        # and moreover, enforces that any overload of __bool__ must return True or False.
        # FWIW, sympy also raises TypeError in this case.
        raise TypeError(
            "Cannot determine truth value of Constraint. "
            "If you are trying to combine Constraint's with logical connectives, "
            "you can specify them separately instead."
        )

    @property
    def serializable_spec(self):
        # We need a serialization compatible format of the constraint so that it
        # can be savedin the graph module w/o breaking the module serialization.
        # The saved constraints will be used directly for the post-exporting pass
        # that converts constraints to runtime assertion. The saved constraints
        # will not be saved in the serialized module.
        # TODO: A better way is needed. Currently we use 't_id' to map the constraint,
        # which is not reliable
        return {
            "t_id": self.t_id,
            "dim": self.dim,
            "min": self.constraint_range.vr.lower,
            "max": self.constraint_range.vr.upper,
            "shared": (
                None
                if self.shared is None
                else {
                    "t_id": self.shared.t_id,
                    "dim": self.shared.dim,
                }
            ),
        }

    def __eq__(self, other):
        if not isinstance(other, Constraint):
            raise TypeError(
                "A dynamic dim can be specified equal only to another dynamic dim. "
                f"Equality with {type(other)} is not supported."
            )
        constraint_range = StrictMinMaxConstraint(
            vr=self.constraint_range.vr & other.constraint_range.vr,
            warn_only=False,
        )
        return _create_constraint(
            self.w_tensor,
            self.t_id,
            self.dim,
            constraint_range,
            shared=_ConstraintTarget(other.w_tensor, other.t_id, other.dim),
        )


def constrain_as_value(symbol, min: Optional[int] = None, max: Optional[int] = None):
    """
    Hint `export()` about the constraint of an intermediate scalar value so that subsequent
    branching behaviors that check on the range of aforementioned scalar value can be
    soundly traced.

    .. warning::
        (Note that if the intermediate scalar value will be used as a shape,
        call `constrain_as_size` API instead.)

    For example, following program can not be traced soundly wihout using
    `constrain_as_value` to give `export()` a hint about which branch to take::

        def fn(x):
            v = x.max().item()
            if v > 1024:
                return x
            else:
                return x * 2

    `export()` would give following error::

        torch._dynamo.exc.UserError: Consider annotating your code using
        torch.export.constrain_as_size() or torch.export().constrain_as_value() APIs.
        It appears that you're trying to get a value out of symbolic int/float whose value
        is data-dependent (and thus we do not know the true value.)  The expression we were
        trying to evaluate is f0 > 1024 (unhinted: f0 > 1024).

    Assuming the actual range of `v` can be between [10, 200], you can add a call to
    `constrain_as_value` in the source code like this::

        def fn(x):
            v = x.max().item()

            # Give export() a hint
            torch.export.constrain_as_value(v, min=10, max=200)

            if v > 1024:
                return x
            else:
                return x * 2

    With the additional hint, `export()` would be able to trace the program correctly by taking
    the `else` branch, resulting in following graph::

        graph():
            %arg0_1 := placeholder[target=arg0_1]

            # v = x.max().item()
            %max_1 := call_function[target=torch.ops.aten.max.default](args = (%arg0_1,))
            %_local_scalar_dense := call_function[target=torch.ops.aten._local_scalar_dense.default](args = (%max_1,))

            # Asserting 10 <= v <= 200
            %ge := call_function[target=operator.ge](args = (%_local_scalar_dense, 10))
            %scalar_tensor := call_function[target=torch.ops.aten.scalar_tensor.default](args = (%ge,))
            %_assert_async := call_function[target=torch.ops.aten._assert_async.msg](
                args = (%scalar_tensor, _local_scalar_dense is outside of inline constraint [10, 200].))
            %le := call_function[target=operator.le](args = (%_local_scalar_dense, 200))
            %scalar_tensor_1 := call_function[target=torch.ops.aten.scalar_tensor.default](args = (%le,))
            %_assert_async_1 := call_function[target=torch.ops.aten._assert_async.msg](
                args = (%scalar_tensor_1, _local_scalar_dense is outside of inline constraint [10, 200].))
            %sym_constrain_range := call_function[target=torch.ops.aten.sym_constrain_range.default](
                args = (%_local_scalar_dense,), kwargs = {min: 10, max: 200})

            # Always taking `else` branch to multiply elements `x` by 2 due to hints above
            %mul := call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 2), kwargs = {})
            return (mul,)


    Args:
        symbol: Intermediate scalar value (int-only now) to apply range constraint on.
        min (Optional[int]): Minimum possible value of given symbol (inclusive)
        max (Optional[int]): Maximum possible value of given symbol (inclusive)

    Returns:
        None

    """
    from torch._export.constraints import constrain_as_value

    return constrain_as_value(symbol, min, max)


def constrain_as_size(symbol, min: Optional[int] = None, max: Optional[int] = None):
    """
    Hint `export()` about the constraint of an intermediate scalar value that
    represents shape of a tensor so that subsequent tensor constructors can be
    traced correctly because many operators need to make assumption about range
    of sizes.

    For example, following program can not be traced soundly wihout using
    `constrain_as_size` to give `export()` a hint about shape ranges::

        def fn(x):
            d = x.max().item()
            return torch.ones(v)

    `export()` would give following error::

        torch._dynamo.exc.Unsupported: guard on data-dependent symbolic int/float

    Assuming the actual range of `d` can be between [3, 10], you can add a call to
    `constrain_as_size` in the source code like this::

        def fn(x):
            d = x.max().item()
            torch.export.constrain_as_size(d, min=3, max=10)
            return torch.ones(d)

    With the additional hint, `export()` would be able to trace the program correctly by taking
    the `else` branch, resulting in following graph::

        graph():
            %arg0_1 := placeholder[target=arg0_1]

            # d = x.max().item()
            %max_1 := call_function[target=torch.ops.aten.max.default](args = (%arg0_1,))
            %_local_scalar_dense := call_function[target=torch.ops.aten._local_scalar_dense.default](args = (%max_1,))

            # Asserting 3 <= d <= 10
            %ge := call_function[target=operator.ge](args = (%_local_scalar_dense, 3))
            %scalar_tensor := call_function[target=torch.ops.aten.scalar_tensor.default](args = (%ge,))
            %_assert_async := call_function[target=torch.ops.aten._assert_async.msg](
                args = (%scalar_tensor, _local_scalar_dense is outside of inline constraint [3, 10].))
            %le := call_function[target=operator.le](args = (%_local_scalar_dense, 10))
            %scalar_tensor_1 := call_function[target=torch.ops.aten.scalar_tensor.default](args = (%le,))
            %_assert_async_1 := call_function[target=torch.ops.aten._assert_async.msg](
                args = (%scalar_tensor_1, _local_scalar_dense is outside of inline constraint [3, 10].))
            %sym_constrain_range_for_size := call_function[target=torch.ops.aten.sym_constrain_range_for_size.default](
                args = (%_local_scalar_dense,), kwargs = {min: 3, max: 10})

            # Constructing new tensor with d
            %full := call_function[target=torch.ops.aten.full.default](
                args = ([%_local_scalar_dense], 1),
                kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})

            ......


    .. warning::
        It is illegal to specify a range that contains 0 and 1. 0/1 values are always specialized
        and can not be part of dynamic range.

    Args:
        symbol: Intermediate scalar value (int-only now) to apply range constraint on.
        min (Optional[int]): Minimum possible value of given symbol (inclusive)
        max (Optional[int]): Maximum possible value of given symbol (inclusive)

    Returns:
        None

    """

    from torch._export.constraints import constrain_as_size

    return constrain_as_size(symbol, min, max)


def dynamic_dim(t: torch.Tensor, index: int):
    """
    `dynamic_dim` constructs a `Constraint` object that describes the dynamism of
    a dimension `index` of tensor `t`. `Constraint` objects should be passed to
    `constraints` argument of `export()`.

    Specifically `dynamic_dim` can be used to express following types of dynamism.

    - Size of a dimension is dynamic and unbounded::

        t0 = torch.rand(2, 3)
        t1 = torch.rand(3, 4)

        # First dimension of t0 can be dynamic size rather than always being static size 2
        constraints = [dynamic_dim(t0, 0)]
        ep = export(fn, (t0, t1), constraints=constraints)

    - Size of a dimension is dynamic with a lower bound::

        t0 = torch.rand(10, 3)
        t1 = torch.rand(3, 4)

        # First dimension of t0 can be dynamic size with a lower bound of 5 (inclusive)
        # Second dimension of t1 can be dynamic size with a lower bound of 2 (exclusive)
        constraints = [
            dynamic_dim(t0, 0) >= 5,
            dynamic_dim(t1, 1) > 2,
        ]
        ep = export(fn, (t0, t1), constraints=constraints)

    - Size of a dimension is dynamic with an upper bound::

        t0 = torch.rand(10, 3)
        t1 = torch.rand(3, 4)

        # First dimension of t0 can be dynamic size with a upper bound of 16 (inclusive)
        # Second dimension of t1 can be dynamic size with a upper bound of 8 (exclusive)
        constraints = [
            dynamic_dim(t0, 0) <= 16,
            dynamic_dim(t1, 1) < 8,
        ]
        ep = export(fn, (t0, t1), constraints=constraints)

    - Size of a dimension is dynamic and it is always equal to size of another dynamic dimension::

        t0 = torch.rand(10, 3)
        t1 = torch.rand(3, 4)

        # Sizes of second dimension of t0 and first dimension are always equal
        constraints = [
            dynamic_dim(t0, 1) == dynamic_dim(t1, 0),
        ]
        ep = export(fn, (t0, t1), constraints=constraints)

    - Mix and match all types above as long as they do not express conflicting requirements

    Args:
        t (torch.Tensor): Example input tensor that have dynamic dimension size(s)
        index (int): Index of dynamic dimension

    Returns:
        A `Constraint` object that describes shape dynamism. It can be passed to `export()` so
        that `export()` does not assume static size of specified tensor, i.e. keeping it dynamic
        as a symbolic size rather than specializing according to size of example tracing input.

    """
    from torch._export import dynamic_dim

    return dynamic_dim(t, index)


def export(
    f: Callable,
    args: Tuple[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    constraints: Optional[List[Constraint]] = None,
) -> ExportedProgram:
    """
    `export()` is a one-shot process for capturing a computation graph from
    a PyTorch program Ahead-of-Time (AOT).

    This function traces a callable (an nn.Module, a function or a method)
    containing PyTorch operations and produces an ExportedProgram. The
    ExportedProgram includes PyTorch operations that perform computations
    equivalent to those in the given nn.Module or callable.

    In specific terms, `export()` traces a function `f` by executing it
    with the provided `args` and `kwargs`. It records the PyTorch operations
    invoked during execution to produce the ExportedProgram.


    **Acceptable input/output types**

    Acceptable types of inputs (for `args` and `kwargs`) and outputs include:

    - Primitive types, i.e. `torch.Tensor`, `int`, `float`, `bool` and `str`.
    - Dataclasses (must be registered with torch._export.utils.register_dataclass_as_pytree_node` first)
    - (Nested) Data structures comprising of `dict`, `list`, `tuple`, `namedtuple` and
      `OrderedDict` containing all above types.


    **What's specialized in the program?**

    1. Non-tensor inputs

    `export()` specializes the traced program based on the values of
    inputs that are not torch.Tensors, ie. `int`, `float`, `bool` and `str`.

    For example::

        from torch.export import export

        def fn(x: torch.Tensor, i: int):
            return x + i

        example_inputs = (torch.rand(2, 2), 1)  # i is set to 1 in example inputs
        ep = export(fn, example_inputs)

    would yield an `ExportedProgram` containing following graph::

        %arg0_1 := placeholder[target=arg0_1]
        %arg1_1 := placeholder[target=arg1_1]
        %add := call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, 1), kwargs = {})
        return (add,)

    Notice that `%add` is computed by adding `%arg0_1` and `1`, which is a
    constant rather than `%arg1_1` because integers are specialized.

    2. Rank and static shapes (not values) of input Tensors

    Rank of a tensor is always specialized and treated as constant. Sizes of
    dimensions are also specialized as constant, i.e. static shapes unless
    specified as dynamic via `dynamic_dim` API, for example::

        from torch.export import export

        def fn(x):
            if x.shape[0] > 5:
                return x + 1
            else:
                return x

        example_inputs = (torch.rand(10, 2))
        ep = export(fn, example_inputs)

    Would produce an `ExportedProgram` containing following graph::

        %arg0_1 := placeholder[target=arg0_1]
        %add := call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, 1), kwargs = {})
        return (add,)

    You can see that the conditional on `x.shape[0]>5` is removed because the
    example inputs has the static shape of `(10, 2)`. `torch.export()` specializes
    on the static shape, thus the `else` branch will never be reached, thus it
    does not show up in the exported program.

    Note:
    If you want to preserve dynamic branching behavior based on value or
    shape of torch.Tensor in the generated graph, you will need to use
    `torch.export.dynamic_dim` to make a dimension of input tensor to be dynamic
    and rewrite the source code using control flow operations like
    `torch.ops.higher_order.cond`.

    3. Control flow

    By default, control flow (like `if`) branching decisions are spcialized
    according to execution flow observed during tracing run. See following
    section on how to preserve dynamic control flow

    **How to express Dynamism**

    1. Shape Dynamism

    Because static shape use cases are more dominant, `export()` chooses to
    assume shapes are all static by default unless there are explicit user
    instructions that say otherwise. Specifically, users can use
    `torch.export.dynamic_dim` to give a hint to `export()` about dynamism
    and range of an input tensor dimension.

    2. Dynamic Control Flow

    To preserve dynamic branching behavior of control flows (like `if`), users
    need to rewrite source code of original program to use PyTorch's higher order
    operators (like `torch.ops.higher_order.cond`).


    **Soundness Guarantee**

    While tracing, `export()` takes note of shape-related assumptions
    made by the user program and the underlying PyTorch operator kernels.
    The output ExportedProgram is considered valid only when these
    assumptions hold true.

    There are 2 types of assumptions made during tracing

    - Shapes (not values) of input tensors.
    - Ranges (lower and upper bound) of values extracted from intermediate tensors via `.item()` or direct indexing.


    All assumptions must be validated at graph capture time for `export()`
    to succeed. Specifically:

    - Assumptions on static shapes of input tensors are automatically validated without additional effort.
    - Assumptions on dynamic shape of input tensors require explicit `Input Constraint`
      constructed with `torch.export.dynamic_dim` APIs
    - Assumptions on range of intermediate values require explicit `Inline Constraint`,
      constructed use `constrain_as_size` and `constraint_as_value` APIs.

    If any assumption can not be validated, a fatal error will be raised. When that happens,
    the error message will include suggested code needed to construct necessary
    constraints to validate the assumptions, for example `export()` would suggest
    following code for input constraints::

        def specify_constraints(x):
            return [
                # x:
                dynamic_dim(x, 0),
                dynamic_dim(x, 0) <= 5,
            ]

    This example means the program requires the dim 0 of input `x` to be less
    than or equal to 5 to be valid. You can inspect the constraints needed and
    then copy this exact function into your code to generated needed
    constraints to be passed into `constraints` argument.

    **ExportedProgram Invariants**

    The returned `ExportedProgram` maintains the following invariants:

    - It is guaranteed to be a sound representation of the original
      program.
    - It maintains the exact calling convention of the original program.
    - It contains a `state_dict` that stores the `torch.nn.Parameters`
      involved in computation of the original program.
    - It includes an fx.GraphModule that represents the computation of
      the original program. The GraphModule:

     - Contains only `placeholder`, `call_function`, `get_attr` and `return` nodes.
     - Inlines all submodules from the original programs.
     - Lifts all parameters and buffers of the original program as inputs to the graph.
     - Does not mutate intermediate values, parameters, or buffers.
     - Does not include operations with side effects.
     - Contains only a curated subset of ATen operations and registered
       custom operations (by default). See the list of Core ATen Ops
       here: https://pytorch.org/docs/stable/ir.html

    Args:
        f: The callable to trace.

        args: Example positional inputs.

        kwargs: Optional example keyword inputs.

        constraints: An optional list of constraints on the dynamic arguments
         that specify their possible range of shapes. By default, shapes of
         input torch.Tensors are assumed to be static. If an input torch.Tensor
         is expected to have dynamic shapes, please use `torch.export.dynamic_dim()`
         to define `Constraint` objects that specify the dynamics and the possible
         range of shapes. See torch.export.dynamic_dim() docstring for examples on
         how to use it.

    Returns:
        An ExportedProgram containing the traced callable.

    """

    from torch._export import export

    return export(f, args, kwargs, constraints)


def save(
    ep: ExportedProgram,
    f: Union[str, pathlib.Path, io.BytesIO],
    *,
    extra_files: Optional[Dict[str, Any]] = None,
    opset_version: Optional[Dict[str, int]] = None,
) -> None:
    """

    .. warning::
        Under active development, saved files may not be usable in newer versions
        of PyTorch.

    Saves an :class:`ExportedProgram` to a file-like object. It can then be
    loaded using the Python API :func:`torch.export.load <torch.export.load>`.

    Args:
        ep (ExportedProgram): The exported program to save.

        f (Union[str, pathlib.Path, io.BytesIO): A file-like object (has to
         implement write and flush) or a string containing a file name.

        extra_files (Optional[Dict[str, Any]]): Map from filename to contents
         which will be stored as part of f.

        opset_version (Optional[Dict[str, int]]): A map of opset names
         to the version of this opset


    Example::

        import torch
        import io

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 10

        ep = torch.export.export(MyModule(), torch.randn(5))

        # Save to file
        torch.export.save(ep, 'exported_program.pt2')

        # Save to io.BytesIO buffer
        buffer = io.BytesIO()
        torch.export.save(ep, buffer)

        # Save with extra files
        extra_files = {'foo.txt': b'bar'}
        torch.export.save(ep, 'exported_program.pt2', extra_files=extra_files)

    """
    from torch._export import save

    save(ep, f, extra_files=extra_files, opset_version=opset_version)


def load(
    f: Union[str, pathlib.Path, io.BytesIO],
    *,
    extra_files: Optional[Dict[str, Any]] = None,
    expected_opset_version: Optional[Dict[str, int]] = None,
) -> ExportedProgram:
    """

    .. warning::
        Under active development, saved files may not be usable in newer versions
        of PyTorch.

    Loads an :class:`ExportedProgram` previously saved with
    :func:`torch.export.save <torch.export.save>`.

    Args:
        ep (ExportedProgram): The exported program to save.

        f (Union[str, pathlib.Path, io.BytesIO): A file-like object (has to
         implement write and flush) or a string containing a file name.

        extra_files (Optional[Dict[str, Any]]): The extra filenames given in
         this map would be loaded and their content would be stored in the
         provided map.

        expected_opset_version (Optional[Dict[str, int]]): A map of opset names
         to expected opset versions

    Returns:
        An :class:`ExportedProgram` object

    Example::

        import torch
        import io

        # Load ExportedProgram from file
        ep = torch.export.load('exported_program.pt2')

        # Load ExportedProgram from io.BytesIO object
        with open('exported_program.pt2', 'rb') as f:
            buffer = io.BytesIO(f.read())
        buffer.seek(0)
        ep = torch.export.load(buffer)

        # Load with extra files.
        extra_files = {'foo.txt': ''}  # values will be replaced with data
        ep = torch.export.load('exported_program.pt2', extra_files=extra_files)
        print(extra_files['foo.txt'])

    """
    from torch._export import load

    return load(
        f, extra_files=extra_files, expected_opset_version=expected_opset_version
    )
