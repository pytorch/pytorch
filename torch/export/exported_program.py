import copy
import dataclasses
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
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
        dialect: Optional[str] = None,
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
    def dialect(self):
        return self._dialect

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch._export.error as error
        from torch._export import combine_args_kwargs

        if self.call_spec.in_spec is not None:
            try:
                user_args = combine_args_kwargs(args, kwargs)
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
        self._check_input_constraints(*ordered_params, *ordered_buffers, *args)

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
                raise error.InternalError(  # noqa: TRY200
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
        from torch._export.passes.replace_sym_size_ops_pass import _ReplaceSymSizeOpPass
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
        new_placeholders = _get_placeholders(gm)
        assert len(new_placeholders) == len(old_placeholders)
        old_new_placeholder_map = {
            old_node.name: new_node.name
            for old_node, new_node in zip(old_placeholders, new_placeholders)
        }
        old_outputs = list(self.graph.nodes)[-1].args[0]
        new_outputs = list(gm.graph.nodes)[-1].args[0]
        assert len(new_outputs) == len(old_outputs)
        old_new_output_map = {
            old_node.name: new_node.name
            for old_node, new_node in zip(old_outputs, new_outputs)
        }

        def make_argument_spec(old_node, node) -> ArgumentSpec:
            if "val" not in node.meta:
                assert len(node.users) == 0
                val = old_node.meta["val"]
            else:
                val = node.meta["val"]
            if isinstance(val, torch.Tensor):
                return TensorArgument(name=node.name)
            elif isinstance(val, torch.SymInt):
                return SymIntArgument(name=node.name)
            else:
                return ConstantArgument(value=val)

        input_specs, output_specs = _sig_to_specs(
            user_inputs={
                old_new_placeholder_map[inp] for inp in self.graph_signature.user_inputs
            },
            inputs_to_parameters={
                old_new_placeholder_map[inp]: param
                for inp, param in self.graph_signature.inputs_to_parameters.items()
            },
            inputs_to_buffers={
                old_new_placeholder_map[inp]: buffer
                for inp, buffer in self.graph_signature.inputs_to_buffers.items()
            },
            user_outputs={
                old_new_output_map[out] for out in self.graph_signature.user_outputs
            },
            buffer_mutations={
                old_new_output_map[out]: buffer
                for out, buffer in self.graph_signature.buffers_to_mutate.items()
            },
            grad_params={},
            grad_user_inputs={},
            loss_output=None,
            inputs=[
                make_argument_spec(old_placeholders[i], node)
                for i, node in enumerate(gm.graph.nodes)
                if node.op == "placeholder"
            ],
            outputs=[
                make_argument_spec(old_outputs[i], node)
                for i, node in enumerate(
                    pytree.tree_leaves(next(iter(reversed(gm.graph.nodes))).args)
                )
            ],
        )

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

        exported_program = ExportedProgram(
            gm,
            gm.graph,
            new_graph_signature,
            self.state_dict,
            new_range_constraints,
            new_equality_constraints,
            copy.deepcopy(self.module_call_graph),
            self.example_inputs,
            self.dialect,
        )

        if len(new_range_constraints) > 0 or len(new_equality_constraints) > 0:
            exported_program = exported_program._transform(
                _AddRuntimeAssertionsForInlineConstraintsPass(
                    new_range_constraints, new_equality_constraints
                )
            )
        exported_program = lift_constant_tensor_pass(exported_program)

        return exported_program._transform(_ReplaceSymSizeOpPass())

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
            new_graph_inputs = [
                node.name for node in new_gm.graph.nodes if node.op == "placeholder"
            ]
            num_inputs = (
                len(old_signature.parameters)
                + len(old_signature.buffers)
                + len(
                    [
                        s
                        for s in old_signature.input_specs
                        if s.kind == InputKind.USER_INPUT
                    ]
                )
            )

            assert len(new_graph_inputs) == num_inputs, (
                f"Number of input nodes changed from {len(new_graph_inputs)} "
                f"to {num_inputs} after transformation. This transformation "
                "is currently not supported."
            )
            num_param_buffers = len(old_signature.buffers) + len(
                old_signature.parameters
            )
            new_user_inputs = new_graph_inputs[num_param_buffers:]

            output_node = list(new_gm.graph.nodes)[-1]
            assert output_node.op == "output"
            new_graph_outputs = [arg.name for arg in output_node.args[0]]

            assert len(new_graph_outputs) == len(old_signature.buffers_to_mutate) + len(
                [
                    s
                    for s in old_signature.output_specs
                    if s.kind == OutputKind.USER_OUTPUT
                ]
            ), (
                f"Number of output nodes changed from {len(new_graph_outputs)} "
                f"to {len(old_signature.buffers_to_mutate) + len(old_signature.user_outputs)} "
                "after transformation. This transformation is currently not supported."
            )
            new_user_outputs = new_graph_outputs[len(old_signature.buffers_to_mutate) :]

            def make_argument_spec(node) -> ArgumentSpec:
                val = node.meta["val"]
                if isinstance(val, torch.Tensor):
                    return TensorArgument(name=node.name)
                elif isinstance(val, torch.SymInt):
                    return SymIntArgument(name=node.name)
                else:
                    return ConstantArgument(value=val)

            input_specs, output_specs = _sig_to_specs(
                user_inputs=set(new_user_inputs),
                inputs_to_parameters=old_signature.inputs_to_parameters,
                inputs_to_buffers=old_signature.inputs_to_buffers,
                user_outputs=set(new_user_outputs),
                buffer_mutations=old_signature.buffers_to_mutate,
                grad_params={},
                grad_user_inputs={},
                loss_output=None,
                inputs=[
                    make_argument_spec(node)
                    for node in transformed_gm.graph.nodes
                    if node.op == "placeholder"
                ],
                outputs=[
                    make_argument_spec(node)
                    for node in pytree.tree_flatten(
                        next(iter(reversed(transformed_gm.graph.nodes))).args
                    )[0]
                ],
            )
            new_signature = ExportGraphSignature(
                input_specs=input_specs, output_specs=output_specs
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
        from torch._export.verifier import Verifier, verify_exported_program_signature

        verify_exported_program_signature(self)

        verifier = Verifier()
        for gm in self.graph_module.modules():
            if not isinstance(gm, torch.fx.GraphModule):
                continue
            verifier.check_valid(self.graph_module)


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
