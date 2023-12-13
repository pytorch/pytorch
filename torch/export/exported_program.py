import copy
import dataclasses
import functools
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # such as auto-completion in tools like pylance, even when these modules are not explicitly
    # imported in user code.

    import sympy

    from torch.utils._sympy.value_ranges import ValueRanges

import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode

from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager

from .graph_signature import (  # noqa: F401
    _sig_to_specs,
    ArgumentSpec,
    ConstantArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    SymIntArgument,
    TensorArgument,
)


__all__ = [
    "ExportedProgram",
    "ModuleCallEntry",
    "ModuleCallSignature",
]


PassType = Callable[[torch.fx.GraphModule], Optional[PassResult]]


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


def _disable_prexisiting_fake_mode(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with maybe_disable_fake_tensor_mode():
            return fn(*args, **kwargs)

    return wrapper


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
        state_dict: Dict[str, Union[torch.Tensor, torch.nn.Parameter]],
        range_constraints: "Dict[sympy.Symbol, Any]",
        equality_constraints: List[Tuple[Any, Any]],
        module_call_graph: List[ModuleCallEntry],
        example_inputs: Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]] = None,
        verifier: Optional[Type[Any]] = None,  # TODO Change typing hint to Verifier.
        tensor_constants: Optional[Dict[str, torch.Tensor]] = None,
    ):
        from torch._export.exported_program import _create_graph_module_for_export
        from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
            InputDim,
        )

        # Remove codegen related things from the graph. It should just be a flat graph.
        graph._codegen = torch.fx.graph.CodeGen()
        self._graph_module = _create_graph_module_for_export(root, graph)
        if isinstance(root, torch.fx.GraphModule):
            self._graph_module.meta.update(root.meta)

        self._graph_signature: ExportGraphSignature = graph_signature
        self._state_dict: Dict[str, Any] = state_dict
        self._range_constraints: "Dict[sympy.Symbol, ValueRanges]" = range_constraints
        self._equality_constraints: List[
            Tuple[InputDim, InputDim]
        ] = equality_constraints
        self._module_call_graph: List[ModuleCallEntry] = module_call_graph
        self._example_inputs = example_inputs

        self._tensor_constants = tensor_constants or {}

        from torch._export.verifier import Verifier

        if verifier is None:
            verifier = Verifier
        assert issubclass(verifier, Verifier)
        self._verifier = verifier
        # Validate should be always the last step of the constructor.
        self.verifier().check(self)

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
    @compatibility(is_backward_compatible=False)
    def call_spec(self):
        from torch._export.exported_program import CallSpec

        if len(self.module_call_graph) == 0:
            return CallSpec(in_spec=None, out_spec=None)
        assert self.module_call_graph[0].fqn == ""
        return CallSpec(
            in_spec=self.module_call_graph[0].signature.in_spec,
            out_spec=self.module_call_graph[0].signature.out_spec,
        )

    @property
    @compatibility(is_backward_compatible=False)
    def verifier(self) -> Any:
        return self._verifier

    @property
    @compatibility(is_backward_compatible=False)
    def dialect(self) -> str:
        return self._verifier.dialect

    @property
    @compatibility(is_backward_compatible=False)
    def tensor_constants(self):
        return self._tensor_constants

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch._export.error as error

        if self.call_spec.in_spec is not None:
            try:
                user_args = (args, kwargs or {})
                args = fx_pytree.tree_flatten_spec(
                    user_args, self.call_spec.in_spec, exact_structural_match=True
                )  # type: ignore[assignment]
            except Exception:
                _, received_spec = pytree.tree_flatten(user_args)
                raise TypeError(  # noqa: TRY200
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
        if hasattr(self.graph_signature, "lifted_tensor_constants"):
            ordered_tensor_constants = tuple(
                self.tensor_constants[name]
                for name in self.graph_signature.lifted_tensor_constants
            )
        else:
            ordered_tensor_constants = ()
        self._check_input_constraints(
            *ordered_params, *ordered_buffers, *ordered_tensor_constants, *args
        )

        # NOTE: calling convention is first params, then buffers, then args as user supplied them.
        # See: torch/_functorch/aot_autograd.py#L1034
        res = torch.fx.Interpreter(self.graph_module).run(
            *ordered_params,
            *ordered_buffers,
            *ordered_tensor_constants,
            *args,
            enable_io_processing=False,
        )

        if self.call_spec.out_spec is not None:
            buffer_mutation = self.graph_signature.buffers_to_mutate
            user_input_mutation = self.graph_signature.user_inputs_to_mutate
            num_mutated = len(buffer_mutation) + len(user_input_mutation)
            mutated_values = res[:num_mutated]

            # Exclude dependency token from final result.
            assertion_dep_token = self.graph_signature.assertion_dep_token
            if assertion_dep_token is not None:
                assertion_dep_token_index = next(iter(assertion_dep_token.keys()))
                res = res[:assertion_dep_token_index]

            res = res[num_mutated:]
            try:
                res = pytree.tree_unflatten(res, self.call_spec.out_spec)
            except Exception:
                _, received_spec = pytree.tree_flatten(res)
                raise error.InternalError(  # noqa: TRY200
                    "Trying to flatten user outputs with exported output tree spec: \n"
                    f"{self.call_spec.out_spec}\n"
                    "but actually got outputs with tree spec of: \n"
                    f"{received_spec}"
                )
            finally:
                user_inputs = [
                    spec
                    for spec in self.graph_signature.input_specs
                    if spec.kind == InputKind.USER_INPUT
                ]
                for i, value in enumerate(mutated_values):
                    output_spec = self.graph_signature.output_specs[i]
                    if output_spec.kind == OutputKind.BUFFER_MUTATION:
                        assert output_spec.target is not None
                        self.state_dict[output_spec.target] = value
                    elif output_spec.kind == OutputKind.USER_INPUT_MUTATION:
                        assert output_spec.target is not None
                        index = next(
                            i
                            for i, spec in enumerate(user_inputs)
                            if spec.arg.name == output_spec.target
                        )
                        args[index].copy_(value)
                    else:
                        raise AssertionError(f"Unexpected kind: {output_spec.kind}")

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

    def module(self, *, flat: bool = True) -> torch.nn.Module:
        """
        Returns a self contained GraphModule with all the parameters/buffers inlined.
        """
        from torch._export.exported_program import unlift_exported_program_lifted_states
        from torch._export.unflatten import unflatten

        if flat:
            return unlift_exported_program_lifted_states(self)
        else:
            return unflatten(self)

    @_disable_prexisiting_fake_mode
    def run_decompositions(
        self, decomp_table: Optional[Dict[torch._ops.OperatorBase, Callable]] = None
    ) -> "ExportedProgram":
        """
        Run a set of decompositions on the exported program and returns a new
        exported program. By default we will run the Core ATen decompositions to
        get operators in the
        `Core ATen Operator Set <https://pytorch.org/docs/stable/torch.compiler_ir.html>`_.

        For now, we do not decompose joint graphs.
        """
        from torch._decomp import core_aten_decompositions
        from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
            _AddRuntimeAssertionsForInlineConstraintsPass,
            InputDim,
        )
        from torch._export.passes.lift_constant_tensor_pass import (
            lift_constant_tensor_pass,
        )
        from torch._export.passes.replace_sym_size_ops_pass import (
            _replace_sym_size_ops_pass,
        )
        from torch._functorch.aot_autograd import aot_export_module

        def _get_placeholders(gm):
            placeholders = []
            for node in gm.graph.nodes:
                if node.op != "placeholder":
                    break
                placeholders.append(node)
            return placeholders

        decomp_table = decomp_table or core_aten_decompositions()

        old_placeholders = _get_placeholders(self.graph_module)
        fake_args = [node.meta["val"] for node in old_placeholders]

        buffers_to_remove = [name for name, _ in self.graph_module.named_buffers()]
        for name in buffers_to_remove:
            delattr(self.graph_module, name)
        # TODO(zhxhchen17) Return the new graph_signature directly.
        gm, graph_signature = aot_export_module(
            self.graph_module, fake_args, decompositions=decomp_table, trace_joint=False
        )

        # Update the signatures with the new placeholder names in case they
        # changed when calling aot_export
        def update_arg(old_arg, new_ph):
            if isinstance(old_arg, ConstantArgument):
                return old_arg
            elif isinstance(old_arg, TensorArgument):
                return TensorArgument(name=new_ph.name)
            elif isinstance(old_arg, SymIntArgument):
                return SymIntArgument(name=new_ph.name)
            raise RuntimeError(f"Type of old_arg not supported: {type(old_arg)}")

        new_placeholders = _get_placeholders(gm)
        new_outputs = list(gm.graph.nodes)[-1].args[0]

        input_specs = [
            InputSpec(spec.kind, update_arg(spec.arg, new_placeholders[i]), spec.target)
            for i, spec in enumerate(self.graph_signature.input_specs)
        ]
        output_specs = [
            OutputSpec(spec.kind, update_arg(spec.arg, new_outputs[i]), spec.target)
            for i, spec in enumerate(self.graph_signature.output_specs)
        ]

        assert len(new_placeholders) == len(old_placeholders)
        old_new_placeholder_map = {
            old_node.name: new_node.name
            for old_node, new_node in zip(old_placeholders, new_placeholders)
        }

        new_graph_signature = ExportGraphSignature(
            input_specs=input_specs, output_specs=output_specs
        )
        # NOTE: aot_export adds symint metadata for placeholders with int
        # values; since these become specialized, we replace such metadata with
        # the original values.
        # Also, set the param/buffer metadata back to the placeholders.
        for old_node, new_node in zip(old_placeholders, new_placeholders):
            if not isinstance(old_node.meta["val"], torch.Tensor):
                new_node.meta["val"] = old_node.meta["val"]

            if (
                new_node.target in new_graph_signature.inputs_to_parameters
                or new_node.target in new_graph_signature.inputs_to_buffers
            ):
                for k, v in old_node.meta.items():
                    new_node.meta[k] = v

        # TODO unfortunately preserving graph-level metadata is not
        # working well with aot_export. So we manually copy it.
        # (The node-level meta is addressed above.)
        gm.meta.update(self.graph_module.meta)

        new_range_constraints = _get_updated_range_constraints(gm)

        new_equality_constraints = [
            (
                InputDim(old_new_placeholder_map[inp_dim1.input_name], inp_dim1.dim),
                InputDim(old_new_placeholder_map[inp_dim2.input_name], inp_dim2.dim),
            )
            for inp_dim1, inp_dim2 in self.equality_constraints
        ]

        lift_constant_tensor_pass(gm, new_graph_signature)
        _replace_sym_size_ops_pass(gm)
        exported_program = ExportedProgram(
            gm,
            gm.graph,
            new_graph_signature,
            self.state_dict,
            new_range_constraints,
            new_equality_constraints,
            copy.deepcopy(self.module_call_graph),
            self.example_inputs,
            self.verifier,
            self.tensor_constants,
        )

        if len(new_range_constraints) > 0 or len(new_equality_constraints) > 0:
            exported_program = exported_program._transform(
                _AddRuntimeAssertionsForInlineConstraintsPass(
                    new_range_constraints, new_equality_constraints
                )
            )

        return exported_program

    def _transform(self, *passes: PassType) -> "ExportedProgram":
        pm = PassManager(list(passes))
        res = pm(self.graph_module)
        transformed_gm = res.graph_module if res is not None else self.graph_module
        assert transformed_gm is not None

        if transformed_gm is self.graph_module and not res.modified:
            return self

        # TODO(zhxchen17) Remove this.
        def _get_updated_graph_signature(
            old_signature: ExportGraphSignature,
            new_gm: torch.fx.GraphModule,
        ) -> ExportGraphSignature:
            """
            Update the graph signature's user_input/user_outputs.
            """
            new_input_specs = []
            for i, node in enumerate(new_gm.graph.nodes):
                if node.op != "placeholder":
                    break

                assert i < len(
                    old_signature.input_specs
                ), "Number of inputs changed after transformation"
                old_input_spec = old_signature.input_specs[i]
                arg = (
                    old_input_spec.arg
                    if isinstance(old_input_spec.arg, ConstantArgument)
                    else type(old_input_spec.arg)(node.name)
                )
                new_input_specs.append(
                    InputSpec(old_input_spec.kind, arg, old_input_spec.target)
                )

            output_node = list(new_gm.graph.nodes)[-1]
            assert output_node.op == "output"

            new_output_specs = []
            for i, node in enumerate(output_node.args[0]):
                assert i < len(
                    old_signature.output_specs
                ), "Number of outputs changed after transformation"
                old_output_spec = old_signature.output_specs[i]
                arg = (
                    old_output_spec.arg
                    if isinstance(old_output_spec.arg, ConstantArgument)
                    else type(old_output_spec.arg)(node.name)
                )
                new_output_specs.append(
                    OutputSpec(old_output_spec.kind, arg, old_output_spec.target)
                )

            new_signature = ExportGraphSignature(
                input_specs=new_input_specs, output_specs=new_output_specs
            )
            return new_signature

        transformed_ep = ExportedProgram(
            transformed_gm,
            transformed_gm.graph,
            _get_updated_graph_signature(self.graph_signature, transformed_gm),
            self.state_dict,
            _get_updated_range_constraints(transformed_gm),
            copy.deepcopy(self.equality_constraints),
            copy.deepcopy(self._module_call_graph),
            self.example_inputs,
            self.verifier,
            self.tensor_constants,
        )
        transformed_ep.graph_module.meta.update(self.graph_module.meta)
        transformed_ep.graph_module.meta.update(res.graph_module.meta)
        return transformed_ep

    def _check_input_constraints(self, *args):
        from torch._export.utils import _check_input_constraints_for_graph

        _check_input_constraints_for_graph(
            self.graph, self.range_constraints, self.equality_constraints
        )(*args)

    def _validate(self):
        self.verifier().check(self)


def _get_updated_range_constraints(
    gm: torch.fx.GraphModule,
) -> "Dict[sympy.Symbol, Any]":
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
        k: v
        for k, v in shape_env.var_to_range.items()
        if k not in shape_env.replacements
    }
    return range_constraints
