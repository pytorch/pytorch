import copy
import dataclasses
import functools
import re
import types
import warnings
from collections import namedtuple
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

from torch.fx.immutable_collections import immutable_dict, immutable_list

if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # such as auto-completion in tools like pylance, even when these modules are not explicitly
    # imported in user code.

    import sympy

    from torch.utils._sympy.value_ranges import ValueRanges

import torch
import torch.utils._pytree as pytree
from torch.export._tree_utils import is_equivalent, reorder_kwargs
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode

from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager

from .graph_signature import (  # noqa: F401
    _sig_to_specs,
    ArgumentSpec,
    ConstantArgument,
    CustomObjArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    SymIntArgument,
    TensorArgument,
    TokenArgument,
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


def _fx_collection_equivalence_fn(
    spec1_type: Optional[type],
    spec1_context: pytree.Context,
    spec2_type: Optional[type],
    spec2_context: pytree.Context,
) -> bool:
    """Treat containers and their immutable variants as the same type. Otherwise
    compare as normal.
    """
    if spec1_type is None or spec2_type is None:
        return spec1_type is spec2_type and spec1_context == spec2_context

    if issubclass(spec1_type, (dict, immutable_dict)) and issubclass(
        spec2_type, (dict, immutable_dict)
    ):
        return spec1_context == spec2_context

    if issubclass(spec1_type, (list, immutable_list)) and issubclass(
        spec2_type, (list, immutable_list)
    ):
        return spec1_context == spec2_context

    return spec1_type is spec2_type and spec1_context == spec2_context


def _rename_without_collisions(
    name_map: Dict[str, str],
    orig_name: str,
    name: str,
    is_placeholder: bool = False,
):
    """
    Renames nodes to avoid name collisions, with suffixing.
    name_map: map from original name to new name
    orig_name: mapping key
    name: candidate name (potentially suffixed, e.g. mul_2)
    is_placeholder: if the node is a placeholder, avoid detecting suffix
    """
    if name in name_map.values():
        # non-placeholder nodes may be suffixed with the count
        # instead of adding another suffix, we will try to increment it
        match = re.match(r"(.*)_(\d+)", name)
        if match and not is_placeholder:
            name, n = match.group(1), int(match.group(2))
        else:
            n = 0
        while (dup_name := f"{name}_{n + 1}") in name_map.values():
            n += 1
        name_map[orig_name] = dup_name
    else:
        name_map[orig_name] = name
    return name_map[orig_name]


def _name_hoo_subgraph_placeholders(gm: torch.fx.GraphModule) -> None:
    """
    Propagate placeholder names from the top-level graph into HigherOrderOp subgraphs,
    and handle collisions with non-placeholders by count suffixing.
    Different HOO subgraph types have different input schemas, so we first enumerate them
    and gather the top-level named placeholder nodes.
    """
    # gather all HOO subgraphs and their top-level named placeholder nodes
    subgraph_ph_tuples: List[Tuple[torch.fx.GraphModule, List[torch.fx.Node]]] = []
    for node in gm.graph.nodes:
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.HigherOrderOperator
        ):
            # HOO subgraphs have varying input schemas, so we enumerate them there
            if node.target._name == "cond":
                _, true_graph, false_graph, cond_args = node._args
                subgraph_ph_tuples.append((getattr(gm, true_graph.target), cond_args))
                subgraph_ph_tuples.append((getattr(gm, false_graph.target), cond_args))
            elif node.target._name == "wrap_with_set_grad_enabled":
                subgraph, phs = node._args[1], node._args[2:]
                subgraph_ph_tuples.append((getattr(gm, subgraph.target), phs))
            elif node.target._name == "map_impl":
                body_graph, array, args = node._args
                subgraph_ph_tuples.append(
                    (getattr(gm, body_graph.target), array + args)
                )

    # propagate names
    for subgraph, hoo_phs in subgraph_ph_tuples:
        name_map: Dict[str, str] = {}
        for i, node in enumerate(subgraph.graph.nodes):
            if i < len(hoo_phs):  # placeholder, retain name
                name_map[node.name] = hoo_phs[i].name
                node.name = node.target = hoo_phs[i].name
            else:  # non-placeholder, check for collisions
                node.name = _rename_without_collisions(name_map, node.name, node.name)

        # recurse and recompile
        _name_hoo_subgraph_placeholders(subgraph)
        subgraph.recompile()


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
        module_call_graph: List[ModuleCallEntry],
        example_inputs: Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]] = None,
        verifier: Optional[Type[Any]] = None,  # TODO Change typing hint to Verifier.
        tensor_constants: Optional[
            Dict[str, torch.Tensor]
        ] = None,  # TODO: deprecate this
        constants: Optional[
            Dict[str, Union[torch.Tensor, torch._C.ScriptObject]]
        ] = None,
    ):
        # Remove codegen related things from the graph. It should just be a flat graph.
        graph._codegen = torch.fx.graph.CodeGen()
        self._graph_module = _create_graph_module_for_export(root, graph)
        if isinstance(root, torch.fx.GraphModule):
            self._graph_module.meta.update(root.meta)

        self._graph_signature: ExportGraphSignature = graph_signature
        self._state_dict: Dict[str, Any] = state_dict
        self._range_constraints: "Dict[sympy.Symbol, ValueRanges]" = range_constraints
        assert module_call_graph is not None
        self._module_call_graph: List[ModuleCallEntry] = module_call_graph
        self._example_inputs = example_inputs

        self._constants = tensor_constants or constants or {}
        assert self._constants is not None

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
        non_persistent_buffers = set(self.graph_signature.non_persistent_buffers)
        for buffer_name in self.graph_signature.buffers:
            if buffer_name in non_persistent_buffers:
                yield buffer_name, self.constants[buffer_name]
            else:
                yield buffer_name, self.state_dict[buffer_name]

    @property
    @compatibility(is_backward_compatible=False)
    def range_constraints(self):
        return self._range_constraints

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
        CallSpec = namedtuple("CallSpec", ["in_spec", "out_spec"])

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
        return self._constants

    @property
    @compatibility(is_backward_compatible=False)
    def constants(self):
        return self._constants

    def _get_flat_args_with_check(self, args, kwargs):
        """Flatten args, kwargs using pytree, then, check specs.

        Args:
            args: List[Any] original args passed to __call__
            kwargs: Dict[str, Any] original kwargs passed to __call

        Returns:
            A tuple of (flat_args, received_spec)
            flat_args is flattend args / kwargs
            received_spec is the pytree spec produced while flattening the
            tuple (args, kwargs)
        """
        in_spec = self.call_spec.in_spec
        if in_spec is not None:
            kwargs = reorder_kwargs(kwargs, in_spec)
        flat_args_with_path, received_spec = pytree.tree_flatten_with_path(
            (args, kwargs)
        )  # type: ignore[possibly-undefined]
        self._check_input_constraints(flat_args_with_path)
        flat_args = tuple(x[1] for x in flat_args_with_path)
        return flat_args, received_spec

    def _graph_module_flat_inputs(self, args: Any, kwargs: Any) -> Any:
        """Transform args, kwargs of __call__ to args for graph_module.

        self.graph_module takes stuff from state dict as inputs.
        The invariant is for ep: ExportedProgram is
        ep(args, kwargs) ==
          ep.postprocess(ep.graph_module(ep.graph_module_flat_inputs(args, kwargs)))
        """

        in_spec = self.call_spec.in_spec
        flat_args, received_spec = self._get_flat_args_with_check(args, kwargs)
        if in_spec is not None and not is_equivalent(
            received_spec, in_spec, _fx_collection_equivalence_fn
        ):
            raise ValueError(
                "Trying to flatten user inputs with exported input tree spec: \n"
                f"{in_spec}\n"
                "but actually got inputs with tree spec of: \n"
                f"{received_spec}"
            )

        additional_inputs = []
        for input_ in self.graph_signature.input_specs:
            if input_.kind == InputKind.USER_INPUT:
                continue
            elif input_.kind in (
                InputKind.PARAMETER,
                InputKind.BUFFER,
            ):
                if input_.persistent is False:
                    # This is a non-persistent buffer, grab it from our
                    # constants instead of the state dict.
                    additional_inputs.append(self.constants[input_.target])
                else:
                    additional_inputs.append(self.state_dict[input_.target])
            elif input_.kind in (
                InputKind.CONSTANT_TENSOR,
                InputKind.CUSTOM_OBJ,
            ):
                additional_inputs.append(self.constants[input_.target])
        additional_inputs = tuple(additional_inputs)

        # NOTE: calling convention is first params, then buffers, then args as user supplied them.
        # See: torch/_functorch/aot_autograd.py#L1034
        return additional_inputs + flat_args

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "Unable to call ExportedProgram directly. "
            "You should use `exported_program.module()` instead."
        )

    def _postprocess_graph_module_outputs(self, res, orig_args, orig_kwargs):
        """Process potential mutations to the input.

        Because self.graph_module is functional, so mutations has to be written
        back after execution of graph_module.
        """
        import torch._export.error as error

        flat_args, _ = self._get_flat_args_with_check(orig_args, orig_kwargs)
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
                        flat_args[index].copy_(value)
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
        )
        return string

    def module(self) -> torch.nn.Module:
        """
        Returns a self contained GraphModule with all the parameters/buffers inlined.
        """
        from ._unlift import _unlift_exported_program_lifted_states

        module = _unlift_exported_program_lifted_states(self)

        def _train(self, mode: bool = True):
            raise NotImplementedError("Calling train() is not supported yet.")

        def _eval(self, mode: bool = True):
            raise NotImplementedError("Calling eval() is not supported yet.")

        module.train = types.MethodType(_train, module)  # type: ignore[method-assign]
        module.eval = types.MethodType(_eval, module)  # type: ignore[method-assign]
        return module

    def _num_lifted_params_buffers(self):
        return next(
            (
                i
                for i, s in enumerate(self._graph_signature.input_specs)
                if s.kind == InputKind.USER_INPUT
            ),
            len(self._graph_signature.input_specs),
        )

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
        )
        from torch._export.passes.lift_constants_pass import (
            ConstantAttrMap,
            lift_constants_pass,
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
        from torch.export._trace import _ignore_backend_decomps

        with _ignore_backend_decomps():
            gm, graph_signature = aot_export_module(
                self.graph_module,
                fake_args,
                decompositions=decomp_table,
                trace_joint=False,
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

        # rename the placeholders
        assert len(new_placeholders) == len(old_placeholders)
        for old_ph, new_ph in zip(old_placeholders, new_placeholders):
            new_ph.name = new_ph.target = old_ph.name

        # handle name collisions with newly decomposed graph nodes
        name_map = {ph.name: ph.name for ph in new_placeholders}
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                continue
            node.name = _rename_without_collisions(name_map, node.name, node.name)

        # propagate names to higher order op subgraphs
        _name_hoo_subgraph_placeholders(gm)

        # To match the output target with correct input for input mutations
        # need to find the old to new placeholder map
        old_new_placeholder_map = {
            spec.arg.name: new_placeholders[i].name
            for i, spec in enumerate(self.graph_signature.input_specs)
            if not isinstance(spec.arg, ConstantArgument)
        }

        input_specs = [
            InputSpec(
                spec.kind,
                update_arg(spec.arg, new_placeholders[i]),
                spec.target,
                spec.persistent,
            )
            for i, spec in enumerate(self.graph_signature.input_specs)
        ]
        output_specs = [
            OutputSpec(
                spec.kind,
                update_arg(spec.arg, new_outputs[i]),
                old_new_placeholder_map.get(spec.target, spec.target),
            )
            for i, spec in enumerate(self.graph_signature.output_specs)
        ]

        assert len(new_placeholders) == len(old_placeholders)

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

        new_range_constraints = _get_updated_range_constraints(
            gm,
            self._num_lifted_params_buffers(),
            pytree.tree_leaves(self.example_inputs),
            _is_executorch=False,
        )

        constants = lift_constants_pass(gm, new_graph_signature, ConstantAttrMap())
        for k, v in constants.items():
            assert k not in self.constants
            self.constants[k] = v

        _replace_sym_size_ops_pass(gm)

        if len(new_range_constraints) > 0:
            res = _AddRuntimeAssertionsForInlineConstraintsPass(new_range_constraints)(
                gm
            )
            assert res is not None
            gm = res.graph_module

        exported_program = ExportedProgram(
            root=gm,
            graph=gm.graph,
            graph_signature=new_graph_signature,
            state_dict=self.state_dict,
            range_constraints=new_range_constraints,
            module_call_graph=copy.deepcopy(self.module_call_graph),
            example_inputs=self.example_inputs,
            verifier=self.verifier,
            constants=self.constants,
        )
        return exported_program

    def _transform_do_not_use(self, *passes: PassType) -> "ExportedProgram":
        pm = PassManager(list(passes))
        # Since we abstractly run the passes, we need to disable backend decomp here
        # again.
        from torch.export._trace import _ignore_backend_decomps

        with _ignore_backend_decomps():
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
                    if isinstance(
                        old_input_spec.arg, (ConstantArgument, CustomObjArgument)
                    )
                    else type(old_input_spec.arg)(node.name)
                )
                new_input_specs.append(
                    InputSpec(
                        old_input_spec.kind,
                        arg,
                        old_input_spec.target,
                        old_input_spec.persistent,
                    )
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
                    if isinstance(
                        old_output_spec.arg, (ConstantArgument, CustomObjArgument)
                    )
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
            root=transformed_gm,
            graph=transformed_gm.graph,
            graph_signature=_get_updated_graph_signature(
                self.graph_signature, transformed_gm
            ),
            state_dict=self.state_dict,
            range_constraints=_get_updated_range_constraints(
                transformed_gm,
                self._num_lifted_params_buffers(),
                pytree.tree_leaves(self.example_inputs),
                _is_executorch=False,
            ),
            module_call_graph=copy.deepcopy(self._module_call_graph),
            example_inputs=self.example_inputs,
            verifier=self.verifier,
            constants=self.constants,
        )
        transformed_ep.graph_module.meta.update(self.graph_module.meta)
        transformed_ep.graph_module.meta.update(res.graph_module.meta)
        return transformed_ep

    def _check_input_constraints(self, flat_args_with_path):
        from torch._export.utils import _check_input_constraints_for_graph

        placeholders = [p for p in self.graph.nodes if p.op == "placeholder"]
        input_placeholders = [
            p
            for p, s in zip(placeholders, self.graph_signature.input_specs)
            if s.kind == InputKind.USER_INPUT
        ]
        _check_input_constraints_for_graph(
            input_placeholders, flat_args_with_path, self.range_constraints
        )

    def _validate(self):
        self.verifier().check(self)

    # TODO(zhxchen17) Formalize this.
    def _update(
        self, graph_module, graph_signature, state_dict=None
    ) -> "ExportedProgram":
        return ExportedProgram(
            root=graph_module,
            graph=graph_module.graph,
            graph_signature=graph_signature,
            state_dict=state_dict or self.state_dict,
            range_constraints=copy.deepcopy(self.range_constraints),
            module_call_graph=copy.deepcopy(self._module_call_graph),
            example_inputs=self.example_inputs,
            verifier=self.verifier,
            tensor_constants=self.tensor_constants,
        )


def _get_updated_range_constraints(
    gm: torch.fx.GraphModule,
    num_lifted: Optional[int] = None,
    example_inputs: Optional[List[Any]] = None,
    _is_executorch: bool = True,
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
            return fake_mode.shape_env, fake_mode
        for v in vals:
            if isinstance(v, torch.SymInt):
                return v.node.shape_env, fake_mode

    # FIXME(tmanlaibaatar) Remove this whole branch once https://github.com/pytorch/pytorch/pull/123764
    if _is_executorch:
        assert num_lifted is None
        assert example_inputs is None
        shape_env, _ = get_shape_env(gm)
        if shape_env is None:
            return {}
        range_constraints = {
            k: v
            for k, v in shape_env.var_to_range.items()
            if k not in shape_env.replacements
        }
        # Only when we have an unbacked symint, and it's used as constructor inputs,
        # runtime_var_to_range will make a difference compated to var_to_range.
        # e.g. [2, oo) -> [0, oo)
        for k, v in shape_env.var_to_range.items():
            if k not in shape_env.replacements:
                range_constraints[k] = v
        return range_constraints

    assert num_lifted is not None
    assert example_inputs is not None

    shape_env, fake_mode = get_shape_env(gm)
    if shape_env is None:
        return {}

    from torch.export.dynamic_shapes import _process_constraints

    range_constraints = _process_constraints(fake_mode, gm, num_lifted, example_inputs)

    range_constraints = {
        k: v for k, v in range_constraints.items() if k not in shape_env.replacements
    }
    # Only when we have an unbacked symint, and it's used as constructor inputs,
    # runtime_var_to_range will make a difference compated to var_to_range.
    # e.g. [2, oo) -> [0, oo)
    for k, v in shape_env.var_to_range.items():
        if k not in shape_env.replacements and k not in range_constraints:
            range_constraints[k] = v
    return range_constraints


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
            "run the graph module eagerly using torch.fx.Interpreter."
        )
        gm = torch.fx.GraphModule(root, torch.fx.Graph())
        gm._graph = graph

    return gm
