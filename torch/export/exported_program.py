# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import contextlib
import copy
import dataclasses
import functools
import operator
import types
import warnings
from collections import namedtuple
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    final,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

from torch._library.fake_class_registry import FakeScriptObject
from torch.fx._utils import first_call_function_nn_module_stack
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts


if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # such as auto-completion in tools like pylance, even when these modules are not explicitly
    # imported in user code.

    import sympy

    from torch.utils._sympy.value_ranges import ValueRanges

import torch
import torch.utils._pytree as pytree
from torch._export.utils import (
    _collect_and_set_constant_attrs,
    _collect_param_buffer_metadata,
    _detect_fake_mode_from_gm,
    _name_hoo_subgraph_placeholders,
    _overwrite_signature_for_non_persistent_buffers,
    _populate_param_buffer_metadata_to_new_gm,
    _rename_without_collisions,
)
from torch._export.verifier import Verifier
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.export._tree_utils import is_equivalent, reorder_kwargs
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager

from .graph_signature import (  # noqa: F401
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

    def replace_all_uses_with(self, original_node, new_node):
        for i in self.inputs:
            if i.name == original_node.name:
                i.name = new_node.name
        for o in self.outputs:
            if o.name == original_node.name:
                o.name = new_node.name


@dataclasses.dataclass
class ModuleCallEntry:
    fqn: str
    signature: Optional[ModuleCallSignature] = None


def _disable_prexisiting_fake_mode(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with unset_fake_temporarily():
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


@contextmanager
def _override_decomp_aten_to_variants():
    from torch.export._trace import _override_composite_implicit_decomp

    # Preserve variants of aten::to understanding that they are mutating/aliasing
    # and their CompositeImplicitAutograd kernels will not become NotImplemented.
    # We will later replace them with aten._to_copy when functionalizing.
    with _override_composite_implicit_decomp(
        (torch.ops.aten.to.dtype_layout, torch.ops.aten.to.dtype),
        {},
        safe=False,
    ):
        yield


def _decompose_and_get_gm_with_new_signature_constants(
    ep,
    *,
    decomp_table: Dict[torch._ops.OperatorBase, Callable],
    _preserve_ops: Tuple[torch._ops.OpOverload],
    joint_loss_index: Optional[int],
):
    from torch._functorch.aot_autograd import aot_export_module
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.export._trace import (
        _export_to_aten_ir,
        _fakify_params_buffers,
        _ignore_backend_decomps,
        _override_composite_implicit_decomp,
        _verify_nn_module_stack,
        _verify_placeholder_names,
        _verify_stack_trace,
    )
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    # TODO Merge this path with inference IR decomp, but it will require some additional work
    # so I will leave it for now. T200307782
    if ep.verifier.dialect == "TRAINING":
        mod = ep.module()

        fake_args = []
        for node in mod.graph.nodes:
            if node.op == "placeholder":
                fake_args.append(node.meta["val"])

        fake_args_unwrapped = pytree.tree_unflatten(fake_args, mod._in_spec)
        fake_mode = _detect_fake_mode_from_gm(mod)
        if fake_mode is None:
            fake_mode = FakeTensorMode(shape_env=ShapeEnv(), export=True)

        # Fix the graph output signature to be tuple if scalar
        out_spec = mod._out_spec

        orig_arg_names = mod.graph._codegen.pytree_info.orig_args  # type: ignore[attr-defined]

        # aot_export expect the return type to always be a tuple.
        if out_spec.type not in (list, tuple):
            out_spec = pytree.TreeSpec(tuple, None, [out_spec])

        mod.graph._codegen = _PyTreeCodeGen(
            _PyTreeInfo(
                orig_arg_names,
                mod._in_spec,
                out_spec,
            )
        )

        mod.recompile()

        # the exported module will store constants & non-persistent buffers such that
        # retracing treats them as persistent buffers, so we inform the constants lifting pass
        # and overwrite the new graph signature using the previous program.
        constant_attrs = _collect_and_set_constant_attrs(
            ep.graph_signature, ep.constants, mod
        )

        # get params & buffers after excluding constants
        fake_params_buffers = _fakify_params_buffers(fake_mode, mod)

        params_buffers_to_node_meta = _collect_param_buffer_metadata(mod)

        with _ignore_backend_decomps(), fake_mode, _override_decomp_aten_to_variants():
            aten_export_artifact = _export_to_aten_ir(
                mod,
                # this requires empty kwargs, but not in pytree.flattened format
                (
                    *fake_args_unwrapped[0],
                    *fake_args_unwrapped[1].values(),
                ),
                {},
                fake_params_buffers,
                constant_attrs,
                preserve_ops=_preserve_ops,
                decomp_table=decomp_table,
                _check_autograd_state=False,
            )

        gm = aten_export_artifact.gm
        new_graph_signature = aten_export_artifact.sig

        _populate_param_buffer_metadata_to_new_gm(
            params_buffers_to_node_meta, gm, new_graph_signature
        )

        # overwrite signature for non-persistent buffers
        new_graph_signature = _overwrite_signature_for_non_persistent_buffers(
            ep.graph_signature, new_graph_signature
        )

        _verify_nn_module_stack(gm)
        _verify_stack_trace(gm)
        _verify_placeholder_names(gm, new_graph_signature)

        return _remove_unneccessary_copy_op_pass(gm, new_graph_signature)

    old_placeholders = [
        node for node in ep.graph_module.graph.nodes if node.op == "placeholder"
    ]
    fake_args = [node.meta["val"] for node in old_placeholders]

    buffers_to_remove = [name for name, _ in ep.graph_module.named_buffers()]
    for name in buffers_to_remove:
        delattr(ep.graph_module, name)

    # TODO(zhxhchen17) Return the new graph_signature directly.
    fake_mode = detect_fake_mode(fake_args)
    fake_mode = contextlib.nullcontext() if fake_mode is None else fake_mode
    with _ignore_backend_decomps(), fake_mode, _override_composite_implicit_decomp(
        _preserve_ops,
        decomp_table,
    ):
        gm, graph_signature = aot_export_module(
            ep.graph_module,
            fake_args,
            decompositions=decomp_table,
            trace_joint=True if joint_loss_index is not None else False,
            output_loss_index=joint_loss_index
            if joint_loss_index is not None
            else None,
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

    new_placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
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

    # Run this pass before creating input/output specs, since size-related CSE/DCE might affect output signature.
    # Overwrite output specs afterwards.
    from torch._export.passes._node_metadata_hook import (
        _node_metadata_hook,
        _set_node_metadata_hook,
    )
    from torch._functorch._aot_autograd.input_output_analysis import _graph_output_names

    if not torch._dynamo.config.do_not_emit_runtime_asserts:
        stack_trace = (
            'File "torch/fx/passes/runtime_assert.py", line 24, '
            "in insert_deferred_runtime_asserts"
        )
        shape_env = _get_shape_env(gm)
        if shape_env is not None:
            with _set_node_metadata_hook(
                gm, functools.partial(_node_metadata_hook, stack_trace=stack_trace)
            ):
                insert_deferred_runtime_asserts(
                    gm,
                    shape_env,
                    f"exported program: {first_call_function_nn_module_stack(gm.graph)}",
                    export=True,
                )

    # update output specs
    gm.recompile()
    for i, name in enumerate(_graph_output_names(gm)):
        if isinstance(new_outputs[i], torch.fx.Node):
            new_outputs[i].name = name

    # To match the output target with correct input for input mutations
    # need to find the old to new placeholder map
    old_new_placeholder_map = {
        spec.arg.name: new_placeholders[i].name
        for i, spec in enumerate(ep.graph_signature.input_specs)
        if not isinstance(spec.arg, ConstantArgument)
    }

    input_specs = [
        InputSpec(
            spec.kind,
            update_arg(spec.arg, new_placeholders[i]),
            spec.target,
            spec.persistent,
        )
        for i, spec in enumerate(ep.graph_signature.input_specs)
    ]
    output_specs = [
        OutputSpec(
            spec.kind,
            update_arg(spec.arg, new_outputs[i]),
            old_new_placeholder_map.get(spec.target, spec.target),
        )
        for i, spec in enumerate(ep.graph_signature.output_specs)
    ]

    if joint_loss_index is not None:
        assert graph_signature.backward_signature is not None
        gradients = graph_signature.backward_signature.gradients_to_user_inputs
        assert len(graph_signature.user_inputs) == len(ep.graph_signature.input_specs)
        specs = {
            graph_signature.user_inputs[i]: spec
            for i, spec in enumerate(ep.graph_signature.input_specs)
            if isinstance(spec.arg, TensorArgument)
        }
        for i, node in enumerate(new_outputs[len(output_specs) :]):
            source = gradients[node.name]
            spec = specs[source]  # type: ignore[index]
            if spec.kind == InputKind.PARAMETER:
                kind = OutputKind.GRADIENT_TO_PARAMETER
                target = spec.target
            elif spec.kind == InputKind.USER_INPUT:
                kind = OutputKind.GRADIENT_TO_USER_INPUT
                target = source
            else:
                raise AssertionError(f"Unknown input kind: {spec.kind}")
            output_specs.append(
                OutputSpec(
                    kind,
                    TensorArgument(name=node.name),
                    target,
                )
            )

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
    return gm, new_graph_signature


def _remove_unneccessary_copy_op_pass(
    gm: torch.fx.GraphModule, new_graph_signature: ExportGraphSignature
) -> Tuple[torch.fx.GraphModule, ExportGraphSignature]:
    """
    Removes redundant copy_ node that was introduced due to mutated buffer.
    """
    with gm._set_replace_hook(new_graph_signature.get_replace_hook()):
        for node in gm.graph.nodes:
            if node.op == "output":
                args, _ = pytree.tree_flatten(node.args)
                for out in args:
                    if (
                        isinstance(out, torch.fx.Node)
                        and out.name in new_graph_signature.buffers_to_mutate
                    ):
                        if (
                            out.op == "call_function"
                            and out.target == torch.ops.aten.copy.default
                        ):
                            out.replace_all_uses_with(out.args[1])  # type: ignore[arg-type]
                            gm.graph.erase_node(out)
        gm.recompile()
    return gm, new_graph_signature


def _common_getitem_elimination_pass(
    gm: torch.fx.GraphModule, graph_signature, module_call_graph
):
    with gm._set_replace_hook(graph_signature.get_replace_hook()):
        for module in gm.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue

            node_id: Dict[torch.fx.Node, str] = {}
            getitems: Dict[str, torch.fx.Node] = {}
            for node in list(module.graph.nodes):
                if node.op == "call_function" and node.target == operator.getitem:
                    source, idx = node.args
                    new_id = f"{node_id[source]}.{idx}"
                    if new_id in getitems:
                        node.replace_all_uses_with(getitems[new_id])
                        for entry in module_call_graph:
                            if entry.signature is not None:
                                entry.signature.replace_all_uses_with(
                                    node, getitems[new_id]
                                )
                        module.graph.erase_node(node)
                    else:
                        getitems[new_id] = node
                        node_id[node] = new_id
                else:
                    node_id[node] = node.name


def _decompose_exported_program(
    ep,
    *,
    decomp_table: Dict[torch._ops.OperatorBase, Callable],
    _preserve_ops: Tuple[torch._ops.OpOverload],
    joint_loss_index: Optional[int],
):
    gm, new_graph_signature = _decompose_and_get_gm_with_new_signature_constants(
        ep,
        decomp_table=decomp_table,
        _preserve_ops=_preserve_ops,
        joint_loss_index=joint_loss_index,
    )

    # TODO unfortunately preserving graph-level metadata is not
    # working well with aot_export. So we manually copy it.
    # (The node-level meta is addressed above.)
    gm.meta.update(ep.graph_module.meta)

    new_range_constraints = _get_updated_range_constraints(
        gm,
        ep.range_constraints,
    )

    exported_program = ExportedProgram(
        root=gm,
        graph=gm.graph,
        graph_signature=new_graph_signature,
        state_dict=ep.state_dict,
        range_constraints=new_range_constraints,
        module_call_graph=copy.deepcopy(ep.module_call_graph),
        example_inputs=ep.example_inputs,
        constants=ep.constants,
    )
    return exported_program


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
        constants: Optional[
            Dict[str, Union[torch.Tensor, FakeScriptObject, torch._C.ScriptObject]]
        ] = None,
        *,
        verifiers: Optional[List[Type[Verifier]]] = None,
    ):
        # Remove codegen related things from the graph. It should just be a flat graph.
        graph._codegen = torch.fx.graph.CodeGen()
        self._graph_module = _create_graph_module_for_export(root, graph)
        if isinstance(root, torch.fx.GraphModule):
            self._graph_module.meta.update(root.meta)

        _common_getitem_elimination_pass(
            self._graph_module, graph_signature, module_call_graph
        )
        self._graph_signature: ExportGraphSignature = graph_signature
        self._state_dict: Dict[str, Any] = state_dict
        self._range_constraints: Dict[sympy.Symbol, ValueRanges] = range_constraints
        assert module_call_graph is not None
        self._module_call_graph: List[ModuleCallEntry] = module_call_graph
        self._example_inputs = example_inputs

        self._constants = constants or {}

        verifiers = verifiers or [Verifier]
        assert all(issubclass(v, Verifier) for v in verifiers)
        self._verifiers = verifiers
        # Validate should be always the last step of the constructor.
        self.validate()

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
        return self._verifiers[0]

    @property
    @compatibility(is_backward_compatible=False)
    def dialect(self) -> str:
        assert self._verifiers is not None
        return self._verifiers[0].dialect

    @property
    @compatibility(is_backward_compatible=False)
    def verifiers(self):
        return self._verifiers

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
                raise error.InternalError(  # noqa: B904
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
        graph_module = self.graph_module.print_readable(
            print_output=False, colored=False
        ).replace("\n", "\n    ")
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
        self,
        decomp_table: Optional[Dict[torch._ops.OperatorBase, Callable]] = None,
        _preserve_ops: Tuple[torch._ops.OpOverload, ...] = (),
    ) -> "ExportedProgram":
        """
        Run a set of decompositions on the exported program and returns a new
        exported program. By default we will run the Core ATen decompositions to
        get operators in the
        `Core ATen Operator Set <https://pytorch.org/docs/stable/torch.compiler_ir.html>`_.

        For now, we do not decompose joint graphs.
        """
        from torch._decomp import core_aten_decompositions

        if decomp_table is None:
            decomp_table = core_aten_decompositions()

        return _decompose_exported_program(
            self,
            decomp_table=decomp_table,
            _preserve_ops=_preserve_ops,  # type: ignore[arg-type]
            joint_loss_index=None,
        )

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
                self.range_constraints,
            ),
            module_call_graph=copy.deepcopy(self._module_call_graph),
            example_inputs=self.example_inputs,
            constants=self.constants,
            verifiers=self.verifiers,
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

    @compatibility(is_backward_compatible=False)
    def validate(self):
        self._validate()

    # TODO: remove this
    @final
    def _validate(self):
        assert (
            len(self.verifiers) > 0
        ), "ExportedProgram must have at least one verifier."
        for v in self.verifiers:
            v().check(self)

    # TODO(zhxchen17) Formalize this.
    def _update(
        self, graph_module, graph_signature, *, state_dict=None, verifiers=None
    ) -> "ExportedProgram":
        return ExportedProgram(
            root=graph_module,
            graph=graph_module.graph,
            graph_signature=graph_signature,
            state_dict=state_dict if state_dict is not None else self.state_dict,
            range_constraints=copy.deepcopy(self.range_constraints),
            module_call_graph=copy.deepcopy(self._module_call_graph),
            example_inputs=self.example_inputs,
            constants=self.constants,
            verifiers=verifiers if verifiers is not None else self.verifiers,
        )


def _get_shape_env(gm):
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


def _get_updated_range_constraints(
    gm: torch.fx.GraphModule,
    old_range_constraints: "Optional[Dict[sympy.Symbol, Any]]" = None,
) -> "Dict[sympy.Symbol, Any]":
    assert old_range_constraints is not None

    shape_env = _get_shape_env(gm)
    if shape_env is None:
        return {}

    range_constraints = copy.copy(old_range_constraints)
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
