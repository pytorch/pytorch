# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import contextlib
import copy
import dataclasses
import functools
import operator
import types
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, final, NamedTuple, TYPE_CHECKING

from torch._guards import tracing, TracingContext
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._library.fake_class_registry import FakeScriptObject
from torch._subclasses.fake_impls import (
    _deregister_op_impl,
    _is_op_registered_to_fake_rule,
    register_op_impl,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx._symbolic_trace import _ConstantAttributeType
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
    _build_cache,
    _collect_all_valid_cia_ops,
    _collect_and_set_constant_attrs,
    _collect_param_buffer_metadata,
    _detect_fake_mode_from_gm,
    _fakify_params_buffers,
    _get_decomp_for_cia,
    _is_preservable_cia_op,
    _name_hoo_subgraph_placeholders,
    _override_graph_signature_for_temp_registered_constants,
    _overwrite_signature_for_non_persistent_buffers,
    _populate_param_buffer_metadata_to_new_gm,
    _register_constants_as_buffers,
    _rename_without_collisions,
    _special_op_to_preserve_cia,
    placeholder_naming_pass,
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


PassType = Callable[[torch.fx.GraphModule], PassResult | None]


@dataclasses.dataclass
class ModuleCallSignature:
    inputs: list[ArgumentSpec]
    outputs: list[ArgumentSpec]
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec
    forward_arg_names: list[str] | None = None

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
    signature: ModuleCallSignature | None = None


def _disable_prexisiting_fake_mode(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with unset_fake_temporarily():
            return fn(*args, **kwargs)

    return wrapper


def _fx_collection_equivalence_fn(
    spec1_type: type | None,
    spec1_context: pytree.Context,
    spec2_type: type | None,
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
def _override_composite_implicit_decomp(cia_ops_to_callable):
    # This function overrides CompositeImplicitAutograd decomp for
    # functional composite ops that user specified. Ideally we want to not-decompose
    # ALL composite ops but today's C++ functinalization relies on
    # the fact that it is working with the opset after decomp is run.
    # Hence we can only do it for functional ops. One caveat is that
    # there are some composite ops that lie about their schema (claimed to be
    # functional but not really aka dropout), for these cases, we just decompose.
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


def _split_decomp_table_to_cia_and_python_decomp(
    decomp_table: dict[torch._ops.OperatorBase, Callable],
) -> tuple[dict[torch._ops.OperatorBase, Callable], ...]:
    all_preservable_cia_ops = set(_collect_all_valid_cia_ops())
    cia_ops_to_callable = {}

    for op in list(decomp_table.keys()):
        # TODO we are silently allowing non-safe(non-functional) ops through a crack
        # due to core aten decomp table having non-functional entries. Once we have
        # a tighter check around core aten decomp, we should warn users about them.
        # Tracking issue: (https://github.com/pytorch/pytorch/issues/135759)

        # if it is a valid CIA op we can mess with in export, we check if it is:
        #  1. Has been marked as to be decomposed. Example:
        #        decomp_table = decomp_table_to_core_aten()
        #        del decomp_table[aten.linear]
        #     In this case, user says decompose everything except for aten.linear
        #  2. Has been marked with custom decomp behaviour. Example:
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
            if op_name.startswith("aten"):
                raise AssertionError(
                    f"This should be a custom op, got aten op: {op_name}"
                )
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
    ep: "ExportedProgram",
    *,
    cia_to_decomp: dict[torch._ops.OperatorBase, Callable],
    python_decomp_table: dict[torch._ops.OperatorBase, Callable],
    joint_loss_index: int | None,
    decompose_custom_triton_ops,
):
    from torch._export.passes.lift_constants_pass import _materialize_and_lift_constants
    from torch._functorch.aot_autograd import aot_export_module
    from torch.export._trace import (
        _disable_custom_triton_op_functional_decomposition,
        _export_to_aten_ir,
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

        wrapped_params_buffers = {
            **dict(mod.named_parameters(remove_duplicate=False)),
            **dict(mod.named_buffers(remove_duplicate=False)),
        }

        from torch._functorch._aot_autograd.subclass_parametrization import (
            unwrap_tensor_subclass_parameters,
        )

        # [NOTE] Unwrapping subclasses AOT
        # In torch.compile, the subclass unwrapping/wrapping happen at runtime
        # but at export, this is impossible as it is intended to be run on
        # C++ environment. As a result, we unwrap subclass parameters AOT. After this,
        # ExportedProgram state_dict won't be same as eager model because eager model
        # could have subclass weights while ExportedProgram will have desugared versions.
        # This is fine because run_decompositions is supposed to specialize to post-autograd
        # graph where the subclass desugaring is supposed to happen.
        unwrap_tensor_subclass_parameters(mod)
        unwrapped_params_buffers = {
            **dict(mod.named_parameters(remove_duplicate=False)),
            **dict(mod.named_buffers(remove_duplicate=False)),
        }

        # TODO T204030333
        fake_mode = _detect_fake_mode_from_gm(ep.graph_module)
        if fake_mode is None:
            fake_mode = FakeTensorMode(shape_env=ShapeEnv(), export=True)

        # Fix the graph output signature to be tuple if scalar
        out_spec = mod._out_spec

        if not isinstance(mod.graph._codegen, _PyTreeCodeGen):
            raise AssertionError(
                f"expected mod.graph._codegen to be _PyTreeCodeGen, got {type(mod.graph._codegen)}"
            )
        orig_arg_names = mod.graph._codegen.pytree_info.orig_args

        # aot_export expect the return type to always be a tuple.
        if out_spec is None:
            raise AssertionError("out_spec must not be None")
        if out_spec.type not in (list, tuple):
            out_spec = pytree.treespec_tuple([out_spec])

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

        # When we have a module with constant attributes, AotDispatcher doesn't actually
        # wrap them as functional tensors, because dynamo would have already made it buffer.
        # In non-strict case, however, AotDispatcher can intercept constants, causing it to not
        # functionalize the operators that are operating on constant tensors. Since dynamo already
        # wraps constants as buffers, we temporarily register the constants as buffers and undo this
        # operation after AOTDispatcher is done.
        temp_registered_constants = _register_constants_as_buffers(
            mod, ep.state_dict, ep.graph_signature.non_persistent_buffers
        )

        # get params & buffers after excluding constants
        fake_params_buffers = _fakify_params_buffers(fake_mode, mod)

        params_buffers_to_node_meta = _collect_param_buffer_metadata(mod)

        # TODO (tmanlaibaatar) Ideally run_decomp should just call _non_strict_export
        # but due to special handling of constants as non-persistent buffers make it little
        # difficult. But we should unify this code path together. T206837815
        from torch._export.non_strict_utils import (
            _enable_graph_inputs_of_type_nn_module,
            _fakify_script_objects,
        )

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

        tx = TracingContext(fake_mode)

        with (
            fake_mode,
            _override_composite_implicit_decomp(
                cia_to_decomp,
            ),
            _enable_graph_inputs_of_type_nn_module(ep.example_inputs),
            tracing(tx),
        ):
            retracing_args_unwrapped = pytree.tree_unflatten(
                retracing_args, mod._in_spec
            )
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
                    _prettify_placeholder_names=False,
                    decompose_custom_triton_ops=decompose_custom_triton_ops,
                )

                # aten_export_artifact.constants contains only fake script objects, we need to map them back
                aten_export_artifact.constants = {
                    fqn: (
                        map_fake_to_real[obj]
                        if isinstance(obj, FakeScriptObject)
                        else obj
                    )
                    for fqn, obj in aten_export_artifact.constants.items()
                }

                gm = aten_export_artifact.gm
                new_graph_signature = aten_export_artifact.sig

                # In the previous step, we assume constants as buffers for AOTDispatcher to
                # functianalize properly, so undo that here
                new_graph_signature = (
                    _override_graph_signature_for_temp_registered_constants(
                        new_graph_signature, temp_registered_constants
                    )
                )

                _populate_param_buffer_metadata_to_new_gm(
                    params_buffers_to_node_meta, gm, new_graph_signature
                )

                # overwrite signature for non-persistent buffers
                new_graph_signature = _overwrite_signature_for_non_persistent_buffers(
                    ep.graph_signature, new_graph_signature
                )

                constants = _materialize_and_lift_constants(
                    gm, new_graph_signature, new_fake_constant_attrs
                )

                placeholder_naming_pass(
                    gm,
                    new_graph_signature,
                    patched_mod,
                    new_fake_args,
                    new_fake_kwargs,
                    fake_params_buffers,
                    constants,
                )

        _verify_nn_module_stack(gm)
        _verify_stack_trace(gm)
        _verify_placeholder_names(gm, new_graph_signature)

        gm, new_graph_signature = _remove_unnecessary_copy_op_pass(
            gm, new_graph_signature
        )

        # When we apply parameterization rule to unwrap
        # subclasses, the state dict will now have different
        # desugared parameters. We need to manually filter those
        # and update the ep.state_dict. Ideally, we should just return
        # the state dict of ep.module but ep.module only stores params
        # buffers that participate in forward. If we undo this behavior,
        # it would break some downstream users.
        new_state_dict = {
            **ep.state_dict,
            **{
                name: p
                for name, p in unwrapped_params_buffers.items()
                if name not in wrapped_params_buffers
            },
        }

        for name, p in wrapped_params_buffers.items():
            # Buffers can be persistent/non-persistent
            if name not in new_state_dict:
                if isinstance(p, torch.nn.Parameter):
                    raise AssertionError(
                        f"expected {name!r} not to be a torch.nn.Parameter when not in state_dict"
                    )

            if name in new_state_dict:
                if name not in unwrapped_params_buffers:
                    new_state_dict.pop(name)

        return gm, new_graph_signature, new_state_dict

    old_placeholders = [
        node for node in ep.graph_module.graph.nodes if node.op == "placeholder"
    ]
    fake_args = [node.meta["val"] for node in old_placeholders]

    buffers_to_remove = [name for name, _ in ep.graph_module.named_buffers()]
    for name in buffers_to_remove:
        delattr(ep.graph_module, name)

    # TODO(zhxhchen17) Return the new graph_signature directly.
    fake_mode_det = detect_fake_mode(fake_args)
    fake_mode_ctx = contextlib.nullcontext() if fake_mode_det is None else fake_mode_det  # type: ignore[assignment]
    custom_triton_ops_decomposition_ctx = (
        contextlib.nullcontext
        if decompose_custom_triton_ops
        else _disable_custom_triton_op_functional_decomposition
    )
    with (
        _ignore_backend_decomps(),
        fake_mode_ctx,
        _override_composite_implicit_decomp(cia_to_decomp),
        custom_triton_ops_decomposition_ctx(),
    ):
        gm, graph_signature = aot_export_module(
            ep.graph_module,
            fake_args,
            decompositions=python_decomp_table,
            trace_joint=joint_loss_index is not None,
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
    new_outputs: tuple[torch.fx.Node, ...] = tuple(gm.graph.output_node().args[0])  # type: ignore[arg-type]

    # rename the placeholders
    if len(new_placeholders) != len(old_placeholders):
        raise AssertionError(
            f"new_placeholders length {len(new_placeholders)} does not match old_placeholders length {len(old_placeholders)}"
        )
    for old_ph, new_ph in zip(old_placeholders, new_placeholders):
        new_ph.name = new_ph.target = old_ph.name

    # handle name collisions with newly decomposed graph nodes
    name_map = {}
    find_available: dict[str, int] = defaultdict(int)
    used_names: set[str] = set()
    for ph in new_placeholders:
        name_map[ph.name] = ph.name
        _build_cache(ph.name, find_available, used_names)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue
        node.name = _rename_without_collisions(
            name_map, find_available, used_names, node.name, node.name
        )

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
                gm,
                functools.partial(
                    _node_metadata_hook, metadata={"stack_trace": stack_trace}
                ),
            ):
                insert_deferred_runtime_asserts(
                    gm,
                    shape_env,
                    f"exported program: {first_call_function_nn_module_stack(gm.graph)}",
                    export=True,
                )

    # update output specs
    gm.recompile()
    for output, name in zip(new_outputs, _graph_output_names(gm)):
        if name is not None:
            output.name = name

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

    output_specs = []

    # handle buffer & input mutations; these appear before loss output & gradients
    # (1) ep.graph_signature.input_specs tells us types of inputs
    # (2) graph_signature.user_inputs tells us node input names in order
    # (3) graph_signature.user_inputs_to_mutate tells us buffer & input mutations
    # map (3) -> (2) for input order, -> (1) for input type
    user_inputs_index = {name: i for i, name in enumerate(graph_signature.user_inputs)}
    mutation_names = list(graph_signature.user_inputs_to_mutate.keys())
    expected_names = [node.name for node in new_outputs[: len(mutation_names)]]
    if mutation_names != expected_names:
        raise AssertionError(
            f"mutation_names {mutation_names} does not match expected {expected_names}"
        )
    for output_name, input_name in graph_signature.user_inputs_to_mutate.items():
        i = user_inputs_index[input_name]
        input_spec = ep.graph_signature.input_specs[i]
        if input_spec.kind not in (InputKind.USER_INPUT, InputKind.BUFFER):
            raise AssertionError(
                f"expected input_spec.kind to be USER_INPUT or BUFFER, got {input_spec.kind}"
            )
        output_kind = (
            OutputKind.BUFFER_MUTATION
            if input_spec.kind == InputKind.BUFFER
            else OutputKind.USER_INPUT_MUTATION
        )
        target = (
            input_spec.target
            if input_spec.kind == InputKind.BUFFER
            else input_spec.arg.name
        )
        output_specs.append(
            OutputSpec(
                kind=output_kind,
                arg=TensorArgument(name=output_name),
                target=target,
            )
        )

    # handle actual user outputs
    for i, spec in enumerate(ep.graph_signature.output_specs):
        output_specs.append(
            OutputSpec(
                OutputKind.LOSS_OUTPUT if i == joint_loss_index else spec.kind,
                update_arg(spec.arg, new_outputs[len(mutation_names) + i]),
                old_new_placeholder_map.get(spec.target, spec.target),
            )
        )

    if joint_loss_index is not None:
        if graph_signature.backward_signature is None:
            raise AssertionError(
                "graph_signature.backward_signature must not be None when joint_loss_index is set"
            )
        gradients = graph_signature.backward_signature.gradients_to_user_inputs
        if len(graph_signature.user_inputs) != len(ep.graph_signature.input_specs):
            raise AssertionError(
                f"graph_signature.user_inputs length {len(graph_signature.user_inputs)} does not match "
                f"input_specs length {len(ep.graph_signature.input_specs)}"
            )
        specs = {
            graph_signature.user_inputs[i]: spec
            for i, spec in enumerate(ep.graph_signature.input_specs)
            if isinstance(spec.arg, TensorArgument)
        }
        for node in new_outputs[len(output_specs) :]:
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

    if len(new_placeholders) != len(old_placeholders):
        raise AssertionError(
            f"new_placeholders length {len(new_placeholders)} does not match old_placeholders length {len(old_placeholders)}"
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
    return gm, new_graph_signature, ep.state_dict


def _remove_unnecessary_copy_op_pass(
    gm: torch.fx.GraphModule, new_graph_signature: ExportGraphSignature
) -> tuple[torch.fx.GraphModule, ExportGraphSignature]:
    """
    Removes redundant copy_ node that was introduced due to mutated buffer.
    """
    with gm._set_replace_hook(new_graph_signature.get_replace_hook()):
        for node in gm.graph.nodes:
            if node.op == "output":
                args, _ = pytree.tree_flatten(node.args)
                for out in args:
                    if isinstance(out, torch.fx.Node) and (
                        out.name in new_graph_signature.buffers_to_mutate
                        or out.name in new_graph_signature.parameters_to_mutate
                    ):
                        if (
                            out.op == "call_function"
                            and out.target is torch.ops.aten.copy.default
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

            node_id: dict[torch.fx.Node, str] = {}
            getitems: dict[str, torch.fx.Node] = {}
            for node in list(module.graph.nodes):
                if node.op == "call_function" and node.target is operator.getitem:
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
    old_gm: torch.fx.GraphModule,
    old_graph_signature: ExportGraphSignature,
    gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    old_module_call_graph: list[ModuleCallEntry],
):
    new_module_call_graph = copy.deepcopy(old_module_call_graph)

    old_nodes = {node.name: node for node in old_gm.graph.nodes}

    old_graph_params_buffers = {
        **old_graph_signature.inputs_to_parameters,
        **old_graph_signature.inputs_to_buffers,
    }
    new_graph_params_buffers = {
        **graph_signature.inputs_to_parameters,
        **graph_signature.inputs_to_buffers,
    }

    # use node-level provenance metadata to create a map
    # from old node names to new node names
    provenance: dict[str, str] = {}

    user_input_counter = 0
    old_user_input_names = [
        node.target for node in old_gm.graph.nodes if node.op == "placeholder"
    ]
    old_user_input_names = list(
        filter(
            lambda x: x not in old_graph_params_buffers
            and x not in old_graph_signature.input_tokens,
            old_user_input_names,
        )
    )
    new_user_input_names = [
        node.target for node in gm.graph.nodes if node.op == "placeholder"
    ]

    for node in gm.graph.nodes:
        if history := node.meta.get("from_node", []):
            provenance[history[-1].name] = node.name

        # For params and buffers, we might have applied parameterizaiton rule
        # so that the names might have changed. But for user inputs, we know we
        # must preserve the old name.
        elif node.op == "placeholder":
            if not (
                node.target in new_graph_params_buffers
                or node.target in graph_signature.input_tokens
            ):
                if node.target in new_user_input_names:
                    if not isinstance(node.name, str):
                        raise AssertionError(
                            f"expected node.name to be str, got {type(node.name)}"
                        )
                    old_name = old_user_input_names[user_input_counter]
                    if not isinstance(old_name, str):
                        raise AssertionError(
                            f"expected old_name to be str, got {type(old_name)}"
                        )
                    provenance[old_name] = node.name
                    user_input_counter += 1

    # For all the parameters and buffers, we first see
    # if they are result of parametrizations and if they
    # are, we log them and error later
    old_param_to_desugared = defaultdict(list)
    for name, target in new_graph_params_buffers.items():
        # if the parameters are not parametrized, the naming won't change.
        if not target.startswith("parametrizations."):
            # If we are in strict mode, we can't just reuse the param names
            if name in old_graph_params_buffers:
                provenance[name] = name
        else:
            old_target = ".".join(target.split(".")[1:-1])
            old_param_to_desugared[old_target].append(name)

    # map old names to new names in module call signatures
    for entry in new_module_call_graph:
        signature = entry.signature
        if signature is None:
            continue
        for x in [*signature.inputs, *signature.outputs]:
            # We noticed that submodule is taking subclass as input. we can't
            # preserve signature here.
            if x.name in old_param_to_desugared:
                raise ValueError(
                    f"It looks like {x.name} is a tensor subclass. "
                    f"Preserving submodule that takes subclass parameter is not supported"
                    f" in inference IR because we desugar them, resulting in more tensors"
                )

            if x.name in provenance:
                x.name = provenance[x.name]

            # This can happen when aten.to is called at graph boundaries.
            # Basically aten.to at post-dispatch level can either be copy
            # or alias. In the alias case, we will no-op it so it will
            # disappear from the graph. If we detect such case, we should
            # reuse the input to aten.to as the new input to the submodule.
            # Technically this can happen for other maybe aliasing ops,
            # but aten.to is probably the most common one.
            elif x.name in old_nodes:
                old_node = old_nodes[x.name]
                if old_node.op == "call_function" and old_node.target in [
                    torch.ops.aten.to.dtype_layout,
                    torch.ops.aten.to.device,
                    torch.ops.aten.to.dtype,
                ]:
                    old_target = old_node.args[0].name
                    if old_target not in provenance:
                        raise ValueError(
                            f"It looks like {old_target} is a tensor subclass. "
                            f"Preserving submodule that takes subclass parameter is not supported"
                            f" in inference IR because we desugar them, resulting in more tensors"
                        )

                    x.name = provenance[old_target]

    return new_module_call_graph


def _decompose_exported_program(
    ep,
    *,
    cia_to_decomp: dict[torch._ops.OperatorBase, Callable],
    python_decomp_table: dict[torch._ops.OperatorBase, Callable],
    joint_loss_index: int | None,
    decompose_custom_triton_ops: bool,
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
        decompose_custom_triton_ops=decompose_custom_triton_ops,
    )

    # The signatures of ep.module_call_graph refer to input / output nodes of
    # the original graph module. However, the new graph module may have
    # new nodes due to decompositions. So we need to update these signatures
    # in the decomposed exported program's module_call_graph.
    new_module_call_graph = _get_updated_module_call_graph(
        ep.graph_module,
        ep.graph_signature,
        gm,
        new_graph_signature,
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

    _graph_module: torch.fx.GraphModule
    """The underlying GraphModule containing the exported computation graph."""

    _graph_signature: ExportGraphSignature
    """The signature containing input/output specifications for the graph."""

    _state_dict: dict[str, Any]
    """Dictionary containing parameter and buffer values from the original module."""

    _range_constraints: "dict[sympy.Symbol, ValueRanges]"
    """Symbolic shape constraints for dynamic shapes in the graph."""

    _module_call_graph: list[ModuleCallEntry]
    """Call graph information tracking module hierarchy and signatures."""

    _example_inputs: tuple[tuple[Any, ...], dict[str, Any]] | None
    """Example inputs used during export, stored as (args, kwargs) tuple."""

    _constants: dict[str, _ConstantAttributeType]
    """Dictionary of constant values used in the graph."""

    _verifiers: list[type[Verifier]]
    """List of verifier classes used to validate the exported program."""

    _guards_code: list[str]

    def __init__(
        self,
        root: torch.nn.Module | dict[str, Any],
        graph: torch.fx.Graph,
        graph_signature: ExportGraphSignature,
        state_dict: dict[str, torch.Tensor | torch.nn.Parameter],
        range_constraints: "dict[sympy.Symbol, Any]",
        module_call_graph: list[ModuleCallEntry],
        example_inputs: tuple[tuple[Any, ...], dict[str, Any]] | None = None,
        constants: dict[str, _ConstantAttributeType] | None = None,
        *,
        verifiers: list[type[Verifier]] | None = None,
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
        self._state_dict: dict[str, Any] = state_dict
        self._range_constraints: dict[sympy.Symbol, ValueRanges] = range_constraints
        if module_call_graph is None:
            raise AssertionError("module_call_graph must not be None")
        self._module_call_graph: list[ModuleCallEntry] = module_call_graph
        self._example_inputs = example_inputs

        self._constants = constants or {}

        verifiers = verifiers or [Verifier]
        if not all(issubclass(v, Verifier) for v in verifiers):
            raise AssertionError(
                f"all verifiers must be subclasses of Verifier, got {verifiers}"
            )
        self._verifiers = verifiers
        # Validate should be always the last step of the constructor.
        self.validate()

        self._guards_code = _convert_guards_to_code(self._graph_module)

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
    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
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
    def named_buffers(self) -> Iterator[tuple[str, torch.Tensor]]:
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

        if value is None:
            self._example_inputs = value
            return

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
        class CallSpec(NamedTuple):
            in_spec: pytree.TreeSpec | None
            out_spec: pytree.TreeSpec | None

        if len(self.module_call_graph) == 0:
            return CallSpec(in_spec=None, out_spec=None)
        if self.module_call_graph[0].fqn != "":
            raise AssertionError(
                f"expected first module_call_graph fqn to be empty string, got {self.module_call_graph[0].fqn!r}"
            )
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
        if self._verifiers is None:
            raise AssertionError("_verifiers must not be None")
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
            flat_args is flattened args / kwargs
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

    def __str__(self) -> str:
        graph_module = self.graph_module.print_readable(
            print_output=False, colored=False
        ).replace("\n", "\n    ")
        graph_signature = str(self.graph_signature).replace("\n", "\n    ")
        string = (
            "ExportedProgram:\n"
            f"    {graph_module}\n"
            f"Graph signature: {graph_signature}\n"
            f"Range constraints: {self.range_constraints}\n"
        )
        return string

    def module(self, check_guards=True) -> torch.fx.GraphModule:
        """
        Returns a self contained GraphModule with all the parameters/buffers inlined.

        - When `check_guards=True` (default), a `_guards_fn` submodule is generated
          and a call to a `_guards_fn` submodule is inserted right after placeholders
          in the graph. This module checks guards on inputs.
        - When `check_guards=False`, a subset of these checks are performed by a
          forward pre-hook on the graph module. No `_guards_fn` submodule is generated.

        """
        from ._unlift import _unlift_exported_program_lifted_states

        module = _unlift_exported_program_lifted_states(self, check_guards=check_guards)

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
        decomp_table: dict[torch._ops.OperatorBase, Callable] | None = None,
        decompose_custom_triton_ops: bool = False,
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

        # Note [Separating decomp_table into CIA decomps and non-CIA decomps]
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
            decompose_custom_triton_ops=decompose_custom_triton_ops,
        )

    def _transform_do_not_use(self, *passes: PassType) -> "ExportedProgram":
        pm = PassManager(list(passes))
        # Since we abstractly run the passes, we need to disable backend decomp here
        # again.
        from torch.export._trace import _ignore_backend_decomps

        with _ignore_backend_decomps():
            res = pm(self.graph_module)
        transformed_gm = res.graph_module if res is not None else self.graph_module
        if transformed_gm is None:
            raise AssertionError("transformed_gm must not be None")

        # pyrefly: ignore [missing-attribute]
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

                if i >= len(old_signature.input_specs):
                    raise AssertionError(
                        f"Number of inputs changed after transformation: got index {i} "
                        f"but only {len(old_signature.input_specs)} input_specs"
                    )
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
            if output_node.op != "output":
                raise AssertionError(
                    f"expected last node to have op='output', got {output_node.op!r}"
                )

            new_output_specs = []
            for i, node in enumerate(output_node.args[0]):
                if i >= len(old_signature.output_specs):
                    raise AssertionError(
                        f"Number of outputs changed after transformation: got index {i} "
                        f"but only {len(old_signature.output_specs)} output_specs"
                    )
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
        # pyrefly: ignore [missing-attribute]
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
        if len(self.verifiers) == 0:
            raise AssertionError("ExportedProgram must have at least one verifier.")
        for v in self.verifiers:
            v().check(self)

    # TODO(zhxchen17) Formalize this.
    def _update(
        self,
        graph_module,
        graph_signature,
        *,
        state_dict=None,
        constants=None,
        verifiers=None,
    ) -> "ExportedProgram":
        return ExportedProgram(
            root=graph_module,
            graph=graph_module.graph,
            graph_signature=graph_signature,
            state_dict=state_dict if state_dict is not None else self.state_dict,
            range_constraints=copy.deepcopy(self.range_constraints),
            module_call_graph=copy.deepcopy(self._module_call_graph),
            example_inputs=self.example_inputs,
            constants=constants if constants is not None else self.constants,
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
    old_range_constraints: "dict[sympy.Symbol, Any] | None" = None,
) -> "dict[sympy.Symbol, Any]":
    if old_range_constraints is None:
        raise AssertionError("old_range_constraints must not be None")

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
            "run the graph module eagerly using torch.fx.Interpreter.",
            stacklevel=2,
        )
        gm = torch.fx.GraphModule(root, torch.fx.Graph())
        gm._graph = graph

    return gm


def _convert_guards_to_code(graph_module):
    shape_env = _get_shape_env(graph_module)
    if shape_env is None:
        return []

    local_vars = {
        var
        for var, sources in shape_env.var_to_sources.items()
        if all(
            not isinstance(source, torch._dynamo.source.ConstantSource)
            for source in sources
        )
    }
    py_printer = torch.fx.experimental.symbolic_shapes.ShapeGuardPythonPrinter(
        shape_env.var_to_sources, lambda s: s.name, shape_env.var_to_sources
    )
    ret = [
        py_printer.doprint(guard.expr)
        for guard in shape_env.guards
        if guard.expr.free_symbols.issubset(local_vars)
    ]
    # TODO Figure out how to resolve guards containing weight sizes.
    # This is not a big deal as _guards_code is mostly empty today.
    return [guard for guard in ret if "L['self']" not in guard]
