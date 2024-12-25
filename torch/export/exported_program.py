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

from torch._higher_order_ops.utils import autograd_not_implemented
from torch._library.fake_class_registry import FakeScriptObject
from torch._subclasses.fake_impls import (
    _deregister_op_impl,
    _is_op_registered_to_fake_rule,
    register_op_impl,
)
from torch._subclasses.fake_tensor import FakeTensorMode
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
    _collect_all_valid_cia_ops,
    _collect_and_set_constant_attrs,
    _collect_param_buffer_metadata,
    _detect_fake_mode_from_gm,
    _get_decomp_for_cia,
    _is_preservable_cia_op,
    _name_hoo_subgraph_placeholders,
    _overwrite_signature_for_non_persistent_buffers,
    _populate_param_buffer_metadata_to_new_gm,
    _rename_without_collisions,
    _special_op_to_preserve_cia,
)
from torch._export.verifier import Verifier
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.export._tree_utils import is_equivalent, reorder_kwargs
from torch.export.decomp_utils import CustomDecompTable
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
    SymBoolArgument,
    SymFloatArgument,
    SymIntArgument,
    TensorArgument,
    TokenArgument,
)


__all__ = [
    "ExportedProgram",
    "ModuleCallEntry",
    "ModuleCallSignature",
    "default_decompositions",
]


PassType = Callable[[torch.fx.GraphModule], Optional[PassResult]]


@dataclasses.dataclass
class ModuleCallSignature:
    inputs: List[ArgumentSpec]
    outputs: List[ArgumentSpec]
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec
    forward_arg_names: Optional[List[str]] = None

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


# This list is compiled from DispatchKey.cpp.
# The idea is that we use these keys to override
# CIA decomp in export
_AUTOGRAD_ALIAS_BACKEND_KEYS_TO_OVERRIDE = [
    torch._C.DispatchKey.AutogradCPU,
    torch._C.DispatchKey.AutogradCUDA,
    torch._C.DispatchKey.AutogradMeta,
    torch._C.DispatchKey.AutogradXLA,
    torch._C.DispatchKey.AutogradLazy,
    torch._C.DispatchKey.AutogradIPU,
    torch._C.DispatchKey.AutogradXPU,
    torch._C.DispatchKey.AutogradMPS,
    torch._C.DispatchKey.AutogradHPU,
    torch._C.DispatchKey.AutogradPrivateUse1,
    torch._C.DispatchKey.AutogradPrivateUse2,
    torch._C.DispatchKey.AutogradPrivateUse3,
]


# This list is compiled from DispatchKey.cpp.
# The idea is that we use these keys to add
# python kernels that directly uses default
# CIA decomp
# See NOTE Registering old CIA to Backend kernel
_BACKEND_KEYS_TO_OVERRIDE = [
    torch._C.DispatchKey.CPU,
    torch._C.DispatchKey.CUDA,
    torch._C.DispatchKey.Meta,
    torch._C.DispatchKey.XLA,
    torch._C.DispatchKey.Lazy,
    torch._C.DispatchKey.IPU,
    torch._C.DispatchKey.XPU,
    torch._C.DispatchKey.MPS,
    torch._C.DispatchKey.HPU,
]


@contextmanager
def _override_composite_implicit_decomp(cia_ops_to_callable, safe=True):
    # This function overrides CompositeImplicitAutograd decomp for
    # functional composite ops that user specified. Ideally we want to not-decompose
    # ALL composite ops but today's C++ functinalization relies on
    # the fact that it is working with the opset after decomp is run.
    # Hence we can only do it for functional ops. One caveat is that
    # there are some composite ops that lie about their schema (claimed to be
    # functional but not really aka dropout), for these cases, we just decompose.

    # When safe=False, we will assume that ops_to_preserve can be mutating/aliasing
    # and their usual decompositions need to be shadowed rather than overridden.
    # Thus we will avoid asserting that they are valid to preserve, and will not
    # replace their CompositeImplicitAutograd kernels with NotImplemented.
    # The only current users of this mode are variants of aten::to that we will
    # replace with aten::_to_copy in FunctionalTensorMode.__torch_dispatch__.
    saved_tables = {}
    patched_ops = set()
    for op_overload, decomp_callable in cia_ops_to_callable.items():
        saved_tables[op_overload] = op_overload.py_kernels.copy()
        patched_ops.add(op_overload)
        for override_dispatch_key in _AUTOGRAD_ALIAS_BACKEND_KEYS_TO_OVERRIDE:
            if override_dispatch_key not in op_overload.py_kernels:
                # TODO (tmanlaibaatar)https://github.com/pytorch/pytorch/issues/129430
                op_overload.py_impl(override_dispatch_key)(
                    autograd_not_implemented(op_overload, deferred_error=True)
                )
        # See NOTE: Registering old CIA to Backend kernel
        # It is important that we cache this before we override py_kernels.
        orig_cia_callable = _get_decomp_for_cia(op_overload)
        if torch._C.DispatchKey.CompositeImplicitAutograd in op_overload.py_kernels:
            del op_overload.py_kernels[torch._C.DispatchKey.CompositeImplicitAutograd]

        if safe:
            op_overload.py_impl(torch._C.DispatchKey.CompositeImplicitAutograd)(
                decomp_callable
            )

        # [NOTE] Directly registering fake tensor rule to CIA ops
        # The problem we are facing here is if your CIA custom rule
        # says we want to preserve the op, we will return NotImplemented.
        # Unfortunately, this will invoke meta device tracing in fake tensor
        # resulting in divergent behaviour for CIA kernels that has device based
        # branching (one case is torch.ops.aten.scaled_dot_product.attention)
        # To get around this issue, we register direct fake impl so that we
        # run the kernel before we actually try to decompose the op in FakeTensorMode.
        # Note that is a no-op in most cases, because:
        #   1) In post dispatch tracing, CIA would have already decomposed
        #   2) Most CIA impl are device agnostic.
        def _force_dispatch_to_orig_cia_callable(fake_tensor_mode, op, *args, **kwargs):
            orig_cia_callable = kwargs["original_callable"]
            del kwargs["original_callable"]
            with fake_tensor_mode:
                return orig_cia_callable(*args, **kwargs)

        if not _is_op_registered_to_fake_rule(op_overload):
            register_op_impl(op_overload)(
                functools.partial(
                    _force_dispatch_to_orig_cia_callable,
                    original_callable=orig_cia_callable,
                )
            )

        for key in _BACKEND_KEYS_TO_OVERRIDE:
            if key not in op_overload.py_kernels:
                # [NOTE] Registering old CIA to Backend kernel
                # We always register original CIA behavior to the backend keys kernel
                # The reason is when we are fake tensor prop-ing or executing real kernel,
                # we end up calling an operator on respective backend, which in python dispatcher,
                # will resolve into CIA key. (see resolve_key in torch/_ops.py)
                # As a result, this CIA now will call into the custom user defined
                # CIA which can cause a problem.
                # To make it more concrete, the case we are handling is:
                #  (1) there is a tensor constant we are performing constant propagation
                #      on during tracing
                #  (2) we invoke an op underneath autograd (either because we are below autograd,
                #      or we are tracing in inference mode), so one of the backend keys gets hit
                #  (3) the op we are invoking has a CIA impl that normally runs in eager mode
                #      (and the user wants to tweak this CIA impl during tracing, but during
                #      const-prop we want the original CIA to run
                op_overload.py_impl(key)(orig_cia_callable)

    try:
        yield
    finally:
        for op in patched_ops:
            op.py_kernels.clear()
            op.py_kernels.update(saved_tables[op])
            op._dispatch_cache.clear()
            _deregister_op_impl(op)


@contextmanager
def _override_decomp_aten_to_variants():
    # Preserve variants of aten::to understanding that they are mutating/aliasing
    # and their CompositeImplicitAutograd kernels will not become NotImplemented.
    # We will later replace them with aten._to_copy when functionalizing.
    with _override_composite_implicit_decomp(
        {
            torch.ops.aten.to.dtype_layout: _special_op_to_preserve_cia,
            torch.ops.aten.to.dtype: _special_op_to_preserve_cia,
        },
        safe=False,
    ):
        yield


def _split_decomp_table_to_cia_and_python_decomp(
    decomp_table: Dict[torch._ops.OperatorBase, Callable]
) -> Tuple[Dict[torch._ops.OperatorBase, Callable], ...]:
    all_preservable_cia_ops = set(_collect_all_valid_cia_ops())
    cia_ops_to_callable = {}

    for op in list(decomp_table.keys()):
        # TODO we are silently allowing non-safe(non-functional) ops through a crack
        # due to core aten decomp table having non-functional entries. Once we have
        # a tigher check around core aten decomp, we should warn users about them.
        # Tracking issue: (https://github.com/pytorch/pytorch/issues/135759)

        # if it is a valid CIA op we can mess with in export, we check if it is:
        #  1. Has been marked as to be decomposed. Example:
        #        decomp_table = decomp_table_to_core_aten()
        #        del decomp_table[aten.linear]
        #     In this case, user says decompose everything except for aten.linear
        #  2. Has been marked with custom decomp behavour. Example:
        #        decomp_table = {aten.linear: some_op}
        # For (1), we want to remove all the CIA ops that weren't handled by user as
        # it suggests they are safe to decompose, so we should remove from preservable_list.
        # for (2), we just plumb the custom decomp to AOTDIspatcher.
        # In both cases, we want to remove this CIA op from the decomp_table as it is special
        # handled.
        if op in all_preservable_cia_ops:
            cia_ops_to_callable[op] = decomp_table[op]
            all_preservable_cia_ops.remove(op)
            del decomp_table[op]
        # If it is a custom op, we want to still preserve or do whatever
        # with it if it is a functional CIA. The reason we don't remove
        # from CIA list is because we don't query custom ops.
        elif _is_preservable_cia_op(op):
            op_name = op.name()
            assert not op_name.startswith("aten"), "This should be a custom op"
            cia_ops_to_callable[op] = decomp_table[op]

    # If we reached here, it means user intentionally deleted these CIA ops from
    # decomp table.
    for k in all_preservable_cia_ops:
        cia_ops_to_callable[k] = _special_op_to_preserve_cia

    return cia_ops_to_callable, decomp_table


def default_decompositions() -> "CustomDecompTable":
    """
    This is the default decomposition table which contains decomposition of
    all ATEN operators to core aten opset. Use this API together with
    :func:`run_decompositions()`
    """
    return CustomDecompTable()


def _decompose_and_get_gm_with_new_signature_constants(
    ep,
    *,
    cia_to_decomp: Dict[torch._ops.OperatorBase, Callable],
    python_decomp_table: Dict[torch._ops.OperatorBase, Callable],
    joint_loss_index: Optional[int],
):
    from torch._functorch.aot_autograd import aot_export_module
    from torch.export._trace import (
        _export_to_aten_ir,
        _fakify_params_buffers,
        _ignore_backend_decomps,
        _verify_nn_module_stack,
        _verify_placeholder_names,
        _verify_stack_trace,
    )
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    def _is_joint_ir_decomp(ep, joint_loss_index):
        return (
            joint_loss_index is not None
            or ep.graph_signature.backward_signature is not None
        )

    if not _is_joint_ir_decomp(ep, joint_loss_index):
        mod = ep.module()

        wrapped_params = dict(mod.named_parameters(remove_duplicate=False))

        from torch._functorch._aot_autograd.subclass_parametrization import (
            unwrap_tensor_subclass_parameters,
        )

        # [NOTE] Unwrapping subclasses AOT
        # In torch.compile, the subclass unwrapping/wrapping happen at runtime
        # but at export, this is impossible as it is intented to be run on
        # C++ environment. As a result, we unwrap subclass parameters AOT. After this,
        # ExportedProgram state_dict won't be same as eager model because eager model
        # could have subclass weights while ExportedProgram will have desugared versions.
        # This is fine because run_decompositions is supposed to specialize to post-autograd
        # graph where the subclass desugaring is supposed to happen.
        unwrap_tensor_subclass_parameters(mod)
        unwrapped_params = dict(mod.named_parameters(remove_duplicate=False))

        # TODO T204030333
        fake_mode = _detect_fake_mode_from_gm(ep.graph_module)
        if fake_mode is None:
            fake_mode = FakeTensorMode(shape_env=ShapeEnv(), export=True)
        retracing_args = []
        for node in mod.graph.nodes:
            if node.op == "placeholder":
                if isinstance(node.meta["val"], CustomObjArgument):
                    real_script_obj = None
                    if node.meta["val"].fake_val is None:
                        real_script_obj = ep.constants[node.meta["val"].name]
                    else:
                        real_script_obj = node.meta["val"].fake_val.real_obj
                    retracing_args.append(real_script_obj)
                else:
                    retracing_args.append(node.meta["val"])

        retracing_args_unwrapped = pytree.tree_unflatten(retracing_args, mod._in_spec)
        # Fix the graph output signature to be tuple if scalar
        out_spec = mod._out_spec

        orig_arg_names = mod.graph._codegen.pytree_info.orig_args

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
        _collect_and_set_constant_attrs(ep.graph_signature, ep.constants, mod)

        # get params & buffers after excluding constants
        fake_params_buffers = _fakify_params_buffers(fake_mode, mod)

        params_buffers_to_node_meta = _collect_param_buffer_metadata(mod)

        # TODO (tmanlaibaatar) Ideally run_decomp should just call _non_strict_export
        # but due to special handling of constants as non-persistent buffers make it little
        # diffucult. But we should unify this code path together. T206837815
        from torch._export.non_strict_utils import _fakify_script_objects

        with (
            fake_mode
        ), _override_decomp_aten_to_variants(), _override_composite_implicit_decomp(
            cia_to_decomp,
        ):
            # this requires empty kwargs, but not in pytree.flattened format
            with _fakify_script_objects(
                mod,
                (
                    *retracing_args_unwrapped[0],
                    *retracing_args_unwrapped[1].values(),
                ),
                {},
                fake_mode,
            ) as (
                patched_mod,
                new_fake_args,
                new_fake_kwargs,
                new_fake_constant_attrs,
                map_fake_to_real,
            ):
                aten_export_artifact = _export_to_aten_ir(
                    patched_mod,
                    new_fake_args,
                    new_fake_kwargs,
                    fake_params_buffers,
                    new_fake_constant_attrs,
                    decomp_table=python_decomp_table,
                    _check_autograd_state=False,
                )

                # aten_export_artifact.constants contains only fake script objects, we need to map them back
                aten_export_artifact.constants = {
                    fqn: map_fake_to_real[obj]
                    if isinstance(obj, FakeScriptObject)
                    else obj
                    for fqn, obj in aten_export_artifact.constants.items()
                }

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

        gm, new_graph_signature = _remove_unneccessary_copy_op_pass(
            gm, new_graph_signature
        )

        # When we apply parameterixzation rule to unwrap
        # subclasses, the state dict will now have different
        # desugared parameters. We need to manually filter those
        # and update the ep.state_dict. Ideally, we should just return
        # the state dict of ep.module but ep.module only stores params
        # buffers that participate in forward. If we undo this behaviour,
        # it would break some downstream users.
        for name, p in unwrapped_params.items():
            if name not in wrapped_params:
                ep.state_dict[name] = p

        for name, p in wrapped_params.items():
            assert name in ep.state_dict
            if name not in unwrapped_params:
                ep.state_dict.pop(name)

        return gm, new_graph_signature, ep.state_dict

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
        cia_to_decomp
    ):
        gm, graph_signature = aot_export_module(
            ep.graph_module,
            fake_args,
            decompositions=python_decomp_table,
            trace_joint=True if joint_loss_index is not None else False,
            output_loss_index=(
                joint_loss_index if joint_loss_index is not None else None
            ),
        )
        gm.graph.eliminate_dead_code()

    # Update the signatures with the new placeholder names in case they
    # changed when calling aot_export
    def update_arg(old_arg, new_ph):
        if isinstance(old_arg, ConstantArgument):
            return old_arg
        elif isinstance(old_arg, TensorArgument):
            return TensorArgument(name=new_ph.name)
        elif isinstance(old_arg, SymIntArgument):
            return SymIntArgument(name=new_ph.name)
        elif isinstance(old_arg, SymFloatArgument):
            return SymFloatArgument(name=new_ph.name)
        elif isinstance(old_arg, SymBoolArgument):
            return SymBoolArgument(name=new_ph.name)
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
            OutputKind.LOSS_OUTPUT if i == joint_loss_index else spec.kind,
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
    return gm, new_graph_signature, ep.state_dict


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


def _get_updated_module_call_graph(
    gm: torch.fx.GraphModule,
    old_module_call_graph: List[ModuleCallEntry],
):
    new_module_call_graph = copy.deepcopy(old_module_call_graph)

    # use node-level provenance metadata to create a map
    # from old node names to new node names
    provenance: Dict[str, str] = {}
    for node in gm.graph.nodes:
        if history := node.meta.get("from_node", []):
            provenance[history[-1].name] = node.name

    # map old names to new names in module call signatures
    for entry in new_module_call_graph:
        signature = entry.signature
        if signature is None:
            continue
        for x in [*signature.inputs, *signature.outputs]:
            x.name = provenance.get(x.name, x.name)

    return new_module_call_graph


def _decompose_exported_program(
    ep,
    *,
    cia_to_decomp: Dict[torch._ops.OperatorBase, Callable],
    python_decomp_table: Dict[torch._ops.OperatorBase, Callable],
    joint_loss_index: Optional[int],
):
    (
        gm,
        new_graph_signature,
        state_dict,
    ) = _decompose_and_get_gm_with_new_signature_constants(
        ep,
        cia_to_decomp=cia_to_decomp,
        python_decomp_table=python_decomp_table,
        joint_loss_index=joint_loss_index,
    )

    # The signatures of ep.module_call_graph refer to input / output nodes of
    # the original graph module. However, the new graph module may have
    # new nodes due to decompositions. So we need to update these signatures
    # in the decomposed exported program's module_call_graph.
    new_module_call_graph = _get_updated_module_call_graph(
        gm,
        ep.module_call_graph,
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
        state_dict=state_dict,
        range_constraints=new_range_constraints,
        module_call_graph=new_module_call_graph,
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

    @graph_module.setter
    @compatibility(is_backward_compatible=False)
    def graph_module(self, value):
        raise RuntimeError("Unable to set ExportedProgram's graph_module attribute.")

    @property
    @compatibility(is_backward_compatible=False)
    def graph(self):
        return self.graph_module.graph

    @graph.setter
    @compatibility(is_backward_compatible=False)
    def graph(self, value):
        raise RuntimeError("Unable to set ExportedProgram's graph attribute.")

    @property
    @compatibility(is_backward_compatible=False)
    def graph_signature(self):
        return self._graph_signature

    @graph_signature.setter
    @compatibility(is_backward_compatible=False)
    def graph_signature(self, value):
        raise RuntimeError("Unable to set ExportedProgram's graph_signature attribute.")

    @property
    @compatibility(is_backward_compatible=False)
    def state_dict(self):
        return self._state_dict

    @state_dict.setter
    @compatibility(is_backward_compatible=False)
    def state_dict(self, value):
        raise RuntimeError("Unable to set ExportedProgram's state_dict attribute.")

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

    @range_constraints.setter
    @compatibility(is_backward_compatible=False)
    def range_constraints(self, value):
        raise RuntimeError(
            "Unable to set ExportedProgram's range_constraints attribute."
        )

    @property
    @compatibility(is_backward_compatible=False)
    def module_call_graph(self):
        return self._module_call_graph

    @module_call_graph.setter
    @compatibility(is_backward_compatible=False)
    def module_call_graph(self, value):
        raise RuntimeError(
            "Unable to set ExportedProgram's module_call_graph attribute."
        )

    @property
    @compatibility(is_backward_compatible=False)
    def example_inputs(self):
        return self._example_inputs

    @example_inputs.setter
    @compatibility(is_backward_compatible=False)
    def example_inputs(self, value):
        # This is allowed
        if not (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], tuple)
            and isinstance(value[1], dict)
        ):
            raise ValueError(
                "Example inputs should be a tuple containing example arguments (as "
                "a tuple), and example kwargs (as a dictionary)."
            )

        args, kwargs = value
        from ._unlift import _check_inputs_match

        _check_inputs_match(args, kwargs, self.call_spec.in_spec)

        self._example_inputs = value

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

    @call_spec.setter
    @compatibility(is_backward_compatible=False)
    def call_spec(self, value):
        raise RuntimeError("Unable to set ExportedProgram's call_spec attribute.")

    @property
    @compatibility(is_backward_compatible=False)
    def verifier(self) -> Any:
        return self._verifiers[0]

    @verifier.setter
    @compatibility(is_backward_compatible=False)
    def verifier(self, value):
        raise RuntimeError("Unable to set ExportedProgram's verifier attribute.")

    @property
    @compatibility(is_backward_compatible=False)
    def dialect(self) -> str:
        assert self._verifiers is not None
        return self._verifiers[0].dialect

    @dialect.setter
    @compatibility(is_backward_compatible=False)
    def dialect(self, value):
        raise RuntimeError("Unable to set ExportedProgram's dialect attribute.")

    @property
    @compatibility(is_backward_compatible=False)
    def verifiers(self):
        return self._verifiers

    @verifiers.setter
    @compatibility(is_backward_compatible=False)
    def verifiers(self, value):
        raise RuntimeError("Unable to set ExportedProgram's verifiers attribute.")

    @property
    @compatibility(is_backward_compatible=False)
    def tensor_constants(self):
        return self._constants

    @tensor_constants.setter
    @compatibility(is_backward_compatible=False)
    def tensor_constants(self, value):
        raise RuntimeError(
            "Unable to set ExportedProgram's tensor_constants attribute."
        )

    @property
    @compatibility(is_backward_compatible=False)
    def constants(self):
        return self._constants

    @constants.setter
    @compatibility(is_backward_compatible=False)
    def constants(self, value):
        raise RuntimeError("Unable to set ExportedProgram's constants attribute.")

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
        )
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
    ) -> "ExportedProgram":
        """
        Run a set of decompositions on the exported program and returns a new
        exported program. By default we will run the Core ATen decompositions to
        get operators in the
        `Core ATen Operator Set <https://pytorch.org/docs/stable/torch.compiler_ir.html>`_.

        For now, we do not decompose joint graphs.

        Args:
            decomp_table:
             An optional argument that specifies decomp behaviour for Aten ops
             (1) If None, we decompose to core aten decompositions
             (2) If empty, we don't decompose any operator


        Some examples:

        If you don't want to decompose anything

        .. code-block:: python

            ep = torch.export.export(model, ...)
            ep = ep.run_decompositions(decomp_table={})

        If you want to get a core aten operator set except for certain operator, you can do following:

        .. code-block:: python

            ep = torch.export.export(model, ...)
            decomp_table = torch.export.default_decompositions()
            decomp_table[your_op] = your_custom_decomp
            ep = ep.run_decompositions(decomp_table=decomp_table)
        """
        _decomp_table = (
            default_decompositions() if decomp_table is None else dict(decomp_table)
        )

        if isinstance(_decomp_table, CustomDecompTable):
            _decomp_table = _decomp_table.materialize()

        # Note [Seperating decomp_table into CIA decomps and non-CIA decomps]
        # At this point, we have a decomp_table that contains decomp behaviour for
        # both CIA and post-autograd ops.
        # We need to separate the op into two categories:
        # 1. CIA op: These are the ops that we want to override
        #    CompositeImplicitAutograd decomp for. For them, we need to use _override_composite_implicit_decomp
        #    context manager to plumb it through AOTDispatcher
        # 2. Non-CIA op: These ops are only relevant after AOTDIspatcher runs, so just
        #    checking if they are statically functional is enough.
        # For joint IR case tho, we need to use the old path because we can't register
        # custom decomps this way because we can't use context manager as it installs
        # autograd_error node.
        (
            cia_to_decomp,
            python_decomp_table,
        ) = _split_decomp_table_to_cia_and_python_decomp(_decomp_table)

        return _decompose_exported_program(
            self,
            cia_to_decomp=cia_to_decomp,
            python_decomp_table=python_decomp_table,
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
