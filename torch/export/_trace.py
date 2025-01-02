# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import dataclasses
import functools
import inspect
import logging
import re
import time
import warnings
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch._dynamo
import torch.fx
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.exc import UserError, UserErrorType
from torch._export.db.logging import (
    exportdb_error_message,
    get_class_if_classified_error,
)
from torch._export.non_strict_utils import (
    _fakify_script_objects,
    _gather_constant_attrs,
    _NonStrictTorchFunctionHandler,
    make_constraints,
    make_fake_inputs,
    produce_guards_and_solve_constraints,
)
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass
from torch._export.passes.lift_constants_pass import (
    ConstantAttrMap,
    lift_constants_pass,
    rewrite_script_object_meta,
)
from torch._export.utils import (
    _collect_param_buffer_metadata,
    _compiling_state_context,
    _populate_param_buffer_metadata_to_new_gm,
    _update_gm_meta_if_possible,
    apply_runtime_assertion_pass,
    placeholder_naming_pass,
    placeholder_prefixes,
)
from torch._export.verifier import SpecViolationError
from torch._export.wrappers import _wrap_submodules
from torch._functorch._aot_autograd.input_output_analysis import (
    _graph_input_names,
    _graph_output_names,
)
from torch._functorch._aot_autograd.traced_function_transforms import (
    create_functional_call,
)
from torch._functorch._aot_autograd.utils import (
    create_tree_flattened_fn,
    register_buffer_assignment_hook,
)
from torch._functorch.aot_autograd import (
    _detect_attribute_assignment,
    aot_export_module,
)
from torch._guards import detect_fake_mode
from torch._library.fake_class_registry import FakeScriptObject
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch._utils_internal import log_export_usage
from torch.export._unlift import _check_input_constraints_pre_hook
from torch.export.dynamic_shapes import _check_dynamic_shapes, _combine_args
from torch.export.exported_program import OutputKind
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
    ShapeEnv,
)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.fx.graph_module import _get_attr
from torch.utils._pytree import TreeSpec
from torch.utils._sympy.value_ranges import ValueRangeError

from ._safeguard import AutogradStateOpsFailSafeguard
from .exported_program import (
    _disable_prexisiting_fake_mode,
    ExportedProgram,
    InputKind,
    ModuleCallEntry,
    ModuleCallSignature,
)
from .graph_signature import _convert_to_export_graph_signature, ExportGraphSignature


log = logging.getLogger(__name__)


@dataclasses.dataclass
class ExportDynamoConfig:
    """
    Manage Export-specific configurations of Dynamo.
    """

    allow_rnn: bool = True
    reorderable_logging_functions: Set[Callable] = dataclasses.field(
        default_factory=set
    )
    # Emit runtime asserts after AOTAutograd instead.
    # This isn't really necessary, and isn't much more efficient since the runtime asserts pass does CSE,
    # but if we want to reason more about what guards/runtime asserts to emit,
    # this makes it a bit cleaner to do from the export side. Also no real point in running this twice.
    do_not_emit_runtime_asserts = True


@dataclasses.dataclass
class ATenExportArtifact:
    gm: torch.fx.GraphModule
    sig: ExportGraphSignature
    constants: Dict[
        str,
        Union[
            torch.Tensor,
            FakeScriptObject,
            torch.ScriptObject,
        ],
    ]


@dataclasses.dataclass(frozen=True)
class ExportArtifact:
    aten: ATenExportArtifact
    out_spec: TreeSpec
    fake_mode: FakeTensorMode
    module_call_specs: Dict[str, Dict[str, pytree.TreeSpec]]


DEFAULT_EXPORT_DYNAMO_CONFIG = ExportDynamoConfig()
DEFAULT_EXPORT_DYNAMO_CONFIG.reorderable_logging_functions = {
    logging.critical,
    logging.debug,
    logging.error,
    logging.exception,
    logging.info,
    logging.log,
    logging.warning,
    print,
    warnings.warn,
}


@contextmanager
def _ignore_backend_decomps():
    orig_mkldnn_flag = torch.backends.mkldnn.set_flags(False)
    orig_nnpack_flag = torch.backends.nnpack.set_flags(False)
    try:
        yield
    finally:
        torch.backends.mkldnn.set_flags(*orig_mkldnn_flag)
        torch.backends.nnpack.set_flags(*orig_nnpack_flag)


def _fixup_key(x):
    return "L__self__" + _strip_root(x)


def _strip_root(x):
    if isinstance(x, str) and x.startswith("_export_root"):
        stripped = x[len("_export_root") :]
        return stripped[1:] if stripped.startswith(".") else stripped
    return x


def _rewrite_tracepoint_node(gm: torch.fx.GraphModule):
    """
    In-place modifiy input graph module by replacing the export tracepoint with a new node
    that has the same target and args, but with the _export_root stripped from path.
    """
    for node in gm.graph.nodes:
        if node.target == torch.ops.higher_order._export_tracepoint:
            if "path" in node.kwargs:
                path = _strip_root(node.kwargs["path"])
                with gm.graph.inserting_before(node):
                    new_node = gm.graph.create_node(
                        "call_function",
                        torch.ops.higher_order._export_tracepoint,
                        args=node.args,
                        kwargs={
                            "path": path,
                            "kind": node.kwargs["kind"],
                        },
                    )
                    new_node.meta = node.meta
                    node.replace_all_uses_with(new_node)
                    gm.graph.erase_node(node)


def _extract_fake_inputs(gm, args, kwargs):
    """
    Given a graph module, extract fakified input tensors from the metadata of
    its placeholders, and map them to the structure of given args and kwargs.
    Also return the fake mode used to fakify those inputs.
    """

    fake_inps: List[torch.Tensor] = []
    fake_vals: List[torch.Tensor] = []
    for node in gm.graph.nodes:
        if node.op == "placeholder" and "val" in node.meta:
            fake_val = node.meta["val"]
            if fake_val is not None and isinstance(fake_val, torch.Tensor):
                fake_inps.append(fake_val)
        elif "example_value" in node.meta:
            fake_val = node.meta["example_value"]
            if fake_val is not None and isinstance(fake_val, torch.Tensor):
                fake_vals.append(fake_val)

    if detected_fake_mode := detect_fake_mode(fake_inps + fake_vals):
        fake_mode = detected_fake_mode
    else:
        fake_mode = FakeTensorMode(shape_env=ShapeEnv(), export=True)

    count = 0

    def lookup_fake(x):
        nonlocal count
        val = fake_inps[count]
        count += 1
        return val

    fake_args = pytree.tree_map_only(torch.Tensor, lookup_fake, args)
    fake_kwargs = pytree.tree_map_only(torch.Tensor, lookup_fake, kwargs)

    return fake_args, fake_kwargs, fake_mode


def _replace_param_buffer_names(param_buffer_table, sig):
    for spec in sig.input_specs:
        if spec.kind in (
            InputKind.PARAMETER,
            InputKind.BUFFER,
        ):
            spec.target = param_buffer_table[spec.target]
    for spec in sig.output_specs:
        if spec.kind in (
            OutputKind.BUFFER_MUTATION,
            OutputKind.GRADIENT_TO_PARAMETER,
        ):
            spec.target = param_buffer_table[spec.target]


def _convert_to_positional_args(orig_arg_names, args, kwargs):
    assert len(orig_arg_names) == len(args) + len(kwargs), (
        f"Total number of arg names is expected to be {len(orig_arg_names)} "
        f"but got {len(args)} positional args, {len(kwargs)} kwargs."
    )
    reordered_kwargs = [kwargs[kw_name] for kw_name in orig_arg_names[len(args) :]]
    return (
        *args,
        *reordered_kwargs,
    )


def _normalize_nn_module_stack(gm_torch_level, root_cls):
    # Append a root module to every nn_module_stack.
    root = "L['self']"
    root_key = re.sub(r"[^a-zA-Z0-9]", "_", root)
    for gm in gm_torch_level.modules():
        if not isinstance(gm, torch.fx.GraphModule):
            continue
        for node in gm.graph.nodes:
            if node.op in ["placeholder", "output"]:
                continue
            add_root = True
            if nn_module_stack := node.meta.get("nn_module_stack", {}):
                path, ty = next(iter(nn_module_stack.values()))
                # After deserializing the class `ty` might not exist anymore so
                # it could be a string
                if inspect.isclass(ty) and issubclass(ty, torch.nn.Module):
                    # TODO Figure out why sometimes we have root sometimes we don't.
                    if path == root and ty is root_cls:
                        add_root = False
                else:
                    assert isinstance(ty, str)
            if add_root:

                def normalize_path(path):
                    try:
                        parts = []

                        class Path:
                            def __getattr__(self, name):
                                if name != "_modules":
                                    parts.append(name)
                                return self

                            def __getitem__(self, idx):
                                parts.append(str(idx))
                                return self

                        eval(path, {"L": {"self": Path()}})
                        return ".".join(parts)
                    except Exception:  # TODO(zhxchen17) Remove this.
                        return path

                nn_module_stack = {
                    root_key: (root, root_cls.__module__ + "." + root_cls.__qualname__),
                    **nn_module_stack,
                }
                node.meta["nn_module_stack"] = {
                    key: (normalize_path(path), ty)
                    for key, (path, ty) in nn_module_stack.items()
                }


def _get_param_buffer_mapping(
    original_module: torch.nn.Module,
    traced_module: torch.nn.Module,
) -> Dict[str, str]:
    """
    Returns a mapping of parameter/buffer names from the new module to the
    original model. This is to help with restoring the FQN for parameter/buffers
    of a traced module to what the original module contains.
    """

    param_lookup: Dict[int, str] = {}
    buffer_lookup: Dict[int, str] = {}
    for name, param in original_module.named_parameters(remove_duplicate=False):
        param_lookup[id(param)] = name
    for name, buffer in original_module.named_buffers(remove_duplicate=False):
        buffer_lookup[id(buffer)] = name

    param_buffer_table: Dict[str, str] = {}
    for dynamo_name, dynamo_param in traced_module.named_parameters(
        remove_duplicate=False
    ):
        assert dynamo_name not in param_buffer_table
        if id(dynamo_param) in param_lookup:
            param_buffer_table[dynamo_name] = param_lookup[id(dynamo_param)]

    for dynamo_name, dynamo_buffer in traced_module.named_buffers(
        remove_duplicate=False
    ):
        assert dynamo_name not in param_buffer_table
        if id(dynamo_buffer) in buffer_lookup:
            param_buffer_table[dynamo_name] = buffer_lookup[id(dynamo_buffer)]

    return param_buffer_table


def _preserve_requires_grad_pass(
    gm: torch.fx.GraphModule,
    sig: ExportGraphSignature,
    fake_params_buffers: Dict[str, torch.Tensor],
    constants: Dict[str, Union[torch.Tensor, FakeScriptObject, torch.ScriptObject]],
    flat_fake_args: List[Any],
):
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    assert len(sig.input_specs) == len(placeholders)
    i = 0
    for node, spec in zip(placeholders, sig.input_specs):
        if spec.kind in (
            InputKind.PARAMETER,
            InputKind.BUFFER,
        ):
            assert spec.target is not None
            node.meta["val"].requires_grad = fake_params_buffers[
                spec.target
            ].requires_grad
        elif spec.kind == InputKind.USER_INPUT:
            fake_arg = flat_fake_args[i]
            if isinstance(fake_arg, torch.Tensor):
                node.meta["val"].requires_grad = fake_arg.requires_grad
            i += 1
        elif spec.kind == InputKind.CONSTANT_TENSOR:
            assert spec.target is not None
            constant = constants[spec.target]
            if isinstance(constant, torch.Tensor):
                # If the tensor is not leaf, it should already have a correct requires grad field
                if node.meta["val"].is_leaf:
                    node.meta["val"].requires_grad = constant.requires_grad
                else:
                    assert node.meta["val"].requires_grad == constant.requires_grad
        elif spec.kind in (InputKind.CUSTOM_OBJ, InputKind.TOKEN):
            continue
        else:
            raise AssertionError(spec.kind)


def _remap_constants(
    orig_constant_attrs: ConstantAttrMap,
    graph_signature: ExportGraphSignature,
    constants: Dict[str, Union[torch.Tensor, FakeScriptObject, torch.ScriptObject]],
) -> None:
    """Rewrite the graph signature and constants table to use the FQN from the original module."""
    remap_table: Dict[str, List[str]] = {}
    for name, value in constants.items():
        if value in orig_constant_attrs:
            remap_table[name] = orig_constant_attrs[value]

    for spec in graph_signature.input_specs:
        if spec.kind in (
            InputKind.CONSTANT_TENSOR,
            InputKind.CUSTOM_OBJ,
        ):
            orig_target = spec.target
            assert orig_target is not None
            targets = remap_table.get(orig_target, [orig_target])
            spec.target = targets[0]

            constant = constants[orig_target]
            del constants[orig_target]
            for target in targets:
                constants[target] = constant


def _produce_aten_artifact(
    *,
    gm: torch.fx.GraphModule,
    mod,
    constant_attrs,
    graph_signature,
    pre_dispatch,
    fake_args,
    fake_kwargs,
    fake_params_buffers,
) -> ATenExportArtifact:
    """
    This is a helper function that is shared between export_to_aten_ir and export_to_aten_ir_make_fx
    to produce the aten artifact. (export compatible graph module + signature)

    It does:
    1. Applies runtime assertion pass
    2. Populate meta val when missing
    3. Lift constants as placeholders
    4. Replace raw autograd and autocast ops with HOPs
    5. Prettify names for placeholders
    6. Preserve requires_grad value on node meta val
    """
    # Run runtime asserts pass before creating input/output specs, since size-related CSE/DCE might affect output signature.
    # Overwrite output specs afterwards.
    flat_fake_args = pytree.tree_leaves((fake_args, fake_kwargs))
    gm, graph_signature = apply_runtime_assertion_pass(gm, graph_signature)

    total_non_user_inputs = (
        len(graph_signature.parameters)
        + len(graph_signature.buffers)
        + len(graph_signature.input_tokens)
    )
    set_missing_meta_vals(gm, flat_fake_args, total_non_user_inputs)

    export_graph_signature: Optional[ExportGraphSignature]
    export_graph_signature = _convert_to_export_graph_signature(
        graph_signature, gm, _get_non_persistent_buffers(mod)
    )

    # script objects are always stored in constants no matter whether they're initial inputs or
    # they're lifted in aot" before rewrite_script_object_meta
    constants = rewrite_script_object_meta(gm)
    constants.update(lift_constants_pass(gm, export_graph_signature, constant_attrs))

    if pre_dispatch:
        from torch._export.passes.replace_autocast_with_hop_pass import (
            replace_autocast_with_hop_pass,
        )
        from torch._export.passes.replace_set_grad_with_hop_pass import (
            replace_set_grad_with_hop_pass,
        )

        # Note: replace_set_grad_with_hop_pass need to be after lift_constant_pass because
        # a getattr of a constant tensor doesn't have meta["val"] until after lift_constant_pass.
        # If replace_set_grad_with_hop_pass is before lift_constant_pass,
        # and the constant_tensor is passed as input of the set grad hop, the placeholder's
        # meta["val"] will be None and fails our verifier for placeholder.
        gm, export_graph_signature = replace_set_grad_with_hop_pass(
            gm, export_graph_signature
        )

        gm, export_graph_signature = replace_autocast_with_hop_pass(
            gm, export_graph_signature
        )

    # Remove nn_module_stack, stack_trace metadata from all placeholders/inputs nodes.
    for _mod in gm.modules():
        if not isinstance(_mod, torch.fx.GraphModule):
            continue
        for node in _mod.graph.nodes:
            if node.op in ["placeholder", "output"]:
                node.meta.pop("nn_module_stack", None)
                node.meta.pop("stack_trace", None)

    # Prettify names for placeholder nodes.
    assert export_graph_signature is not None
    placeholder_naming_pass(
        gm,
        export_graph_signature,
        mod,
        fake_args,
        fake_kwargs,
        fake_params_buffers,
        constants,
    )

    _preserve_requires_grad_pass(
        gm, export_graph_signature, fake_params_buffers, constants, flat_fake_args
    )

    return ATenExportArtifact(
        gm,
        export_graph_signature,
        constants,
    )


def _rename_constants_nodes(
    gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
) -> None:
    """
    For strict mode, rename constants nodes that were previously annotated as buffers.
    """
    # handle name collisions with existing constants
    node_names = {node.name for node in gm.graph.nodes}

    def rename_constant(name):
        if name in node_names:
            n = 1
            while (dup_name := f"{name}_{n}") in node_names:
                n += 1
            name = dup_name
        node_names.add(name)
        return name

    # use input specs to map names from buffers to constants
    buffer_prefix = placeholder_prefixes[InputKind.BUFFER]
    const_prefix = placeholder_prefixes[InputKind.CONSTANT_TENSOR]
    buffer_to_constant = {}
    for spec in graph_signature.input_specs:
        if spec.kind == InputKind.CONSTANT_TENSOR and not spec.arg.name.startswith(
            const_prefix
        ):
            if spec.arg.name.startswith(buffer_prefix):  # map from buffer to constants
                c_name = rename_constant(
                    const_prefix + spec.arg.name[len(buffer_prefix) :]
                )
            else:  # lifted constant
                c_name = rename_constant(const_prefix + spec.arg.name)
            buffer_to_constant[spec.arg.name] = c_name
            spec.arg.name = c_name
    for spec in graph_signature.output_specs:
        if spec.arg.name in buffer_to_constant:
            spec.arg.name = buffer_to_constant[spec.arg.name]

    # Rename constants nodes for all modules
    for mod in gm.modules():
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        for node in mod.graph.nodes:
            if node.name in buffer_to_constant:
                node.name = node.target = buffer_to_constant[node.name]
        mod.recompile()


def _restore_state_dict(
    original_module: torch.nn.Module, traced_module: torch.fx.GraphModule
) -> None:
    """
    Restores the state dict of the traced module to that of the original module.
    """
    param_buffer_table = _get_param_buffer_mapping(original_module, traced_module)
    # Since the graph module is flattened (no module heirarchy), we
    # need to noramlize the module by replacing "." with "_". If we
    # don't, it will try to save the weight to a submodule which no
    # longer exists.
    for name, fqn in param_buffer_table.items():
        param_buffer_table[name] = fqn.replace(".", "_")

    # Replace state dict attr names with the fqn
    for name, fqn in param_buffer_table.items():
        if not hasattr(traced_module, name):
            continue

        attr = getattr(traced_module, name)
        if isinstance(attr, torch.Tensor) and not isinstance(attr, torch.nn.Parameter):
            traced_module.register_buffer(fqn, attr)
        else:
            setattr(traced_module, fqn, attr)
        delattr(traced_module, name)

    # Replace graph getattr nodes with the correct name
    for node in traced_module.graph.nodes:
        if node.op == "get_attr":
            attr_name = node.target
            if attr_name in param_buffer_table:
                node.target = param_buffer_table[attr_name]

    traced_module.recompile()


def _get_module_hierarchy(mod: torch.nn.Module) -> Dict[str, str]:
    return {
        name: type(m).__name__ for name, m in mod.named_modules(remove_duplicate=False)
    }


def _make_module_call_graph(
    in_spec: TreeSpec,
    out_spec: TreeSpec,
    module_call_signatures: Dict[str, ModuleCallSignature],
    forward_arg_names: Optional[List[str]] = None,
) -> List[ModuleCallEntry]:
    original = [
        ModuleCallEntry(fqn=fqn, signature=module_call_signatures.get(fqn))
        for fqn in _EXPORT_MODULE_HIERARCHY  # type: ignore[union-attr]
    ]
    assert original[0].fqn == ""
    original[0].signature = ModuleCallSignature(
        inputs=[],
        outputs=[],
        in_spec=in_spec,
        out_spec=out_spec,
        forward_arg_names=forward_arg_names,
    )
    additional = [
        ModuleCallEntry(fqn=fqn, signature=signature)
        for fqn, signature in module_call_signatures.items()
        if fqn not in _EXPORT_MODULE_HIERARCHY  # type: ignore[operator]
    ]
    return [*original, *additional]


def _export_to_torch_ir(
    f: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]] = None,
    *,
    preserve_module_call_signature: Tuple[str, ...] = (),
    disable_constraint_solver: bool = False,
    allow_complex_guards_as_runtime_asserts: bool = False,
    restore_fqn: bool = True,
    _log_export_usage: bool = True,
    same_signature: bool = True,
) -> torch.fx.GraphModule:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a torch.fx.GraphModule in torch IR.
    """

    if _log_export_usage:
        log_export_usage(event="export.private_api", flags={"_export_to_torch_ir"})

    if not isinstance(args, tuple):
        raise UserError(
            UserErrorType.INVALID_INPUT,
            f"Expecting `args` to be a tuple of example positional inputs, got {type(args)}",
        )

    kwargs = kwargs or {}
    combined_args = _combine_args(f, args, kwargs)
    _check_dynamic_shapes(combined_args, dynamic_shapes)
    with torch._dynamo.config.patch(dataclasses.asdict(DEFAULT_EXPORT_DYNAMO_CONFIG)):
        try:
            module_call_specs: Dict[str, Dict[str, pytree.TreeSpec]] = {}
            with _wrap_submodules(
                f, preserve_module_call_signature, module_call_specs
            ), _ignore_backend_decomps():
                gm_torch_level, _ = torch._dynamo.export(
                    f,
                    dynamic_shapes=dynamic_shapes,  # type: ignore[arg-type]
                    assume_static_by_default=True,
                    tracing_mode="symbolic",
                    disable_constraint_solver=disable_constraint_solver,
                    # currently the following 2 flags are tied together for export purposes,
                    # but untangle for sake of dynamo export api
                    prefer_deferred_runtime_asserts_over_guards=True,
                    allow_complex_guards_as_runtime_asserts=allow_complex_guards_as_runtime_asserts,
                    _log_export_usage=_log_export_usage,
                    same_signature=same_signature,
                )(
                    *args,
                    **kwargs,
                )
        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e))  # noqa: B904
        except GuardOnDataDependentSymNode as e:
            raise UserError(  # noqa: B904
                UserErrorType.ANTI_PATTERN,
                f"Consider annotating your code using torch._check*(). {str(e)}",
                case_name="constrain_as_size_example",
            )

    gm_torch_level.meta["module_call_specs"] = module_call_specs

    if isinstance(f, torch.nn.Module) and restore_fqn:
        _restore_state_dict(f, gm_torch_level)

    return gm_torch_level


def _export_to_aten_ir(
    mod: torch.nn.Module,
    fake_args,
    fake_kwargs,
    fake_params_buffers,
    constant_attrs: ConstantAttrMap,
    produce_guards_callback=None,
    *,
    transform=lambda x: x,  # TODO(zhxchen17) Revisit if this is needed later.
    pre_dispatch=False,
    decomp_table=None,
    _check_autograd_state: bool = True,
    _is_torch_jit_trace: bool = False,
) -> ATenExportArtifact:
    # [NOTE] If the user is exporting under training mode, we want to detect if there is any
    # state change in the autograd global state and error. If the user is exporting under inference
    # mode, we don't care. At predispatch level, we don't care about the state change.
    is_grad_enabled = torch._C.is_grad_enabled()
    grad_safe_guard = nullcontext()
    # export_to_aten_ir is called when we decompose the ep into inference IR
    # In that setting, we actually shouldn't check the state change as at this point,
    # because the intention is specalizing to inference.
    if _check_autograd_state:
        if not pre_dispatch and is_grad_enabled:
            grad_safe_guard = AutogradStateOpsFailSafeguard()  # type: ignore[assignment]

    # This _reparametrize_module makes sure inputs and module.params/buffers have the same fake_mode,
    # otherwise aot_export_module will error out because it sees a mix of fake_modes.
    # And we want aot_export_module to use the fake_tensor mode in dynamo to keep the pipeline easy to reason about.
    with torch.nn.utils.stateless._reparametrize_module(
        mod,
        fake_params_buffers,
        tie_weights=True,
        strict=True,
        stack_weights=True,
    ), grad_safe_guard, _ignore_backend_decomps(), _compiling_state_context():  # type: ignore[attr-defined]
        gm, graph_signature = transform(aot_export_module)(
            mod,
            fake_args,
            trace_joint=False,
            pre_dispatch=pre_dispatch,
            decompositions=decomp_table,
            kwargs=fake_kwargs,
        )

    def _maybe_fixup_gm_and_output_node_meta(old_gm, new_gm):
        if isinstance(old_gm, torch.fx.GraphModule):
            if hasattr(old_gm, "meta"):
                new_gm.meta.update(old_gm.meta)
            old_output_node = list(old_gm.graph.nodes)[-1]
            new_output_node = list(new_gm.graph.nodes)[-1]
            assert old_output_node.op == "output" and new_output_node.op == "output"
            # make sure we don't override any meta
            assert len(new_output_node.meta) == 0
            new_output_node.meta.update(old_output_node.meta)

    # TODO unfortunately preserving graph-level metadata and output node's meta
    # is not working well with aot_export. So we manually copy it.
    # (The node-level meta is addressed above.)
    _maybe_fixup_gm_and_output_node_meta(mod, gm)

    # Run produce guards before we handle runtime asserts.
    # This means we run the export solver before the runtime asserts pass.
    # Right now this doesn't mean much - the export solver is only there for suggested fixes,
    # and we won't even get to constraint solving if that's needed.
    # But if in future we want to control what runtime asserts are emitted for export,
    # or rely on produce_guards + solver for some simplification on runtime asserts, this probably makes sense.
    if produce_guards_callback:
        try:
            produce_guards_callback(gm)
        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e))  # noqa: B904

    return _produce_aten_artifact(
        gm=gm,
        mod=mod,
        constant_attrs=constant_attrs,
        graph_signature=graph_signature,
        pre_dispatch=pre_dispatch,
        fake_args=fake_args,
        fake_kwargs=fake_kwargs,
        fake_params_buffers=fake_params_buffers,
    )


def _fakify_params_buffers(
    fake_mode: FakeTensorMode,
    mod: torch.nn.Module,
) -> Dict[str, Union[torch.Tensor, torch.nn.Parameter]]:
    params_buffers = {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }

    faked_params_buffers = {}
    memo: Dict[int, FakeTensor] = {}
    for key, value in params_buffers.items():
        if id(value) in memo:
            fake_tensor = memo[id(value)]
        else:
            fake_tensor = fake_mode.from_tensor(value, static_shapes=True)
            memo[id(value)] = fake_tensor
        faked_params_buffers[key] = fake_tensor
    return faked_params_buffers  # type: ignore[return-value]


def _get_forward_arg_names(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Gets the argument names to forward that are used, for restoring the
    original signature when unlifting the exported program module.
    - Positional args: retain the original argument names, and enumerate
        *args as args_0, args_1, ...
    - Keyword args: retain the original kwarg names in the order specified
        by the user. This order seems to matter for the current state of
        export lifted modules.
    """
    sig = inspect.signature(mod.forward)
    _args = sig.bind_partial(*args).arguments

    names: List[str] = []
    for name, value in _args.items():
        # handle variable number of positional args
        if sig.parameters[name].kind == inspect._ParameterKind.VAR_POSITIONAL:
            names.extend([f"{name}_{i}" for i, _ in enumerate(value)])
        else:
            names.append(name)
    # order of kwargs matters for input spec
    if kwargs:
        names.extend([kwarg for kwarg, _ in kwargs.items()])

    return names


def _get_non_persistent_buffers(mod: torch.nn.Module) -> Set[str]:
    """
    Returns set of non-persistent buffers in a module and its submodules.
    """
    result = set()
    for name, m in mod.named_modules(remove_duplicate=False):
        for b in m._non_persistent_buffers_set:
            result.add(f"{name}.{b}" if name else b)
    return result


def _rewrite_dynamo_tensor_constants(
    orig_mod_buffers: Set[torch.Tensor],
    traced_mod_buffers: Dict[str, torch.Tensor],
    graph_signature: ExportGraphSignature,
    constants: Dict[str, Union[torch.Tensor, FakeScriptObject, torch.ScriptObject]],
) -> None:
    """
    Dynamo erroneously marks tensor attributes on modules as buffers.
    Rewrite them to be tensor constants.
    """
    for spec in graph_signature.input_specs:
        if spec.kind == InputKind.BUFFER:
            assert spec.target is not None
            value = traced_mod_buffers[spec.target]
            if value not in orig_mod_buffers:
                # This was a tensor constant erroneously marked as a buffer.
                # Convert it into a constant in the graph signature, and add its
                # value to the constants table.
                spec.kind = InputKind.CONSTANT_TENSOR
                constants[spec.target] = value  # type: ignore[arg-type]


def _move_non_persistent_buffers_to_tensor_constants(
    orig_mod: torch.nn.Module,
    graph_signature: ExportGraphSignature,
    constants: Dict[str, Union[torch.Tensor, FakeScriptObject, torch.ScriptObject]],
) -> None:
    """
    Moves non-persistent buffers to tensor constants.
    """
    for spec in graph_signature.input_specs:
        if spec.kind == InputKind.BUFFER and not spec.persistent:
            assert spec.target is not None
            assert spec.target not in constants
            constants[spec.target] = orig_mod.get_buffer(spec.target)  # type: ignore[arg-type]


def _verify_nn_module_stack(graph_module: torch.fx.GraphModule) -> None:
    """
    Perform nn_module_stack checks on the graph.
    Current constraints:
        For the top level graph:
        - populated for 'call_function', 'get_attr'
        - None for 'placeholder', 'output'
        For submodule graphs:
        - None for 'placeholder', output'

    TODO(pianpwk): make this a consistent node-level check once nn_module_stack is populated for cond submodules.
    """
    # Check top-level graph for all nodes, all graphs for placeholder & output nodes
    for i, mod in enumerate([graph_module] + list(graph_module.modules())):
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        for node in mod.graph.nodes:
            if node.op in ["call_function", "get_attr"]:
                if i == 0:
                    if (
                        nn_module_stack := node.meta.get("nn_module_stack", None)
                    ) is None:
                        raise SpecViolationError(
                            f"Node {node} of type {node.op} is missing nn_module_stack metadata"
                        )
                    if not all(
                        isinstance(k, str)
                        and isinstance(v, tuple)
                        and len(v) == 2
                        and all(isinstance(x, str) for x in v)
                        for k, v in nn_module_stack.items()
                    ):
                        raise SpecViolationError(
                            f"Node {node} of type {node.op} has incorrect nn_module_stack metadata format"
                            f"expected Dict[str, Tuple[str, str]], but got {nn_module_stack}"
                        )
            elif node.op in ["placeholder", "output"]:
                if node.meta.get("nn_module_stack", None):
                    raise SpecViolationError(
                        f"Node {node} of type {node.op} contains nn_module_stack metadata, this should be None"
                    )


def _verify_stack_trace(graph_module: torch.fx.GraphModule) -> None:
    """
    Perform stack trace checks on the graph.
    Constraints:
        - None or non-empty str for 'call_function', 'get_attr'
        - None for 'placeholder', 'output'
    """
    for mod in [graph_module, *graph_module.modules()]:
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        for node in graph_module.graph.nodes:
            stack_trace = node.meta.get("stack_trace", None)
            if node.op in ["call_function", "get_attr"]:
                if not (stack_trace is None or isinstance(stack_trace, str)):
                    raise SpecViolationError(
                        f"Node {node} of type {node.op} has invalid stack_trace metadata, "
                        f"expected a string or None but instead found: {stack_trace}"
                    )
            elif node.op in ["placeholder", "output"]:
                if stack_trace:
                    raise SpecViolationError(
                        f"Node {node} of type {node.op} contains stack_trace metadata, "
                        f"expected None but instead found: {stack_trace}"
                    )


def _verify_placeholder_names(
    gm: torch.fx.GraphModule, sig: ExportGraphSignature
) -> None:
    """
    Performs a sanity check on the placeholder node names.
    - User input nodes: no restrictions, should match the original forward() signature
    - Params/buffers/constants/custom_obj/token nodes: should start with prefixes defined in <placeholder_prefixes>
    """
    name_to_kind = {spec.arg.name: spec.kind for spec in sig.input_specs}
    for mod in gm.modules():
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        for node in mod.graph.nodes:
            if node.op == "placeholder":
                if node.name not in name_to_kind:
                    continue
                node_kind = name_to_kind[node.name]
                prefix = placeholder_prefixes[node_kind]
                if not node.name.startswith(prefix):
                    raise SpecViolationError(
                        f"Placeholder node name {node.name} does not follow spec for {node_kind}, name should have prefix: {prefix}"
                    )


def get_ep_stats(ep: ExportedProgram) -> Dict[str, Any]:
    op_count = 0
    op_set = set()
    for m in ep.graph_module.modules():
        if not isinstance(m, torch.fx.GraphModule):
            continue
        for node in m.graph.nodes:
            if node.op != "call_function":
                continue
            op_count += 1
            assert hasattr(node.target, "__module__")
            assert hasattr(node.target, "__name__")
            op_set.add(f"{node.target.__module__}.{node.target.__name__}")
    return {"op_count": op_count, "op_set": op_set}


_EXPORT_FLAGS: Optional[Set[str]] = None
_EXPORT_MODULE_HIERARCHY: Optional[Dict[str, str]] = None


def _log_export_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _EXPORT_FLAGS, _EXPORT_MODULE_HIERARCHY
        try:
            start = time.time()
            ep = fn(*args, **kwargs)
            end = time.time()
            log_export_usage(
                event="export.time",
                metrics=end - start,
                flags=_EXPORT_FLAGS,
                **get_ep_stats(ep),
            )
        except Exception as e:
            t = type(e)
            error_type = t.__module__ + "." + t.__qualname__
            case_name = get_class_if_classified_error(e)
            if case_name is not None:
                log.error(exportdb_error_message(case_name))
                log_export_usage(
                    event="export.error.classified",
                    type=error_type,
                    message=str(e),
                    flags=_EXPORT_FLAGS,
                )
            else:
                log_export_usage(
                    event="export.error.unclassified",
                    type=error_type,
                    message=str(e),
                    flags=_EXPORT_FLAGS,
                )
            raise e
        finally:
            _EXPORT_FLAGS = None
            _EXPORT_MODULE_HIERARCHY = None

        return ep

    return wrapper


def _process_jit_trace_inputs_for_export(example_inputs, example_kwarg_inputs):
    if not isinstance(example_inputs, (tuple, list, dict)):
        example_inputs = (example_inputs,)

    elif isinstance(example_inputs, list):
        example_inputs = tuple(example_inputs)

    elif (
        isinstance(example_inputs, (torch.Tensor, dict))
        and example_kwarg_inputs is None
    ):
        example_inputs = (example_inputs,)

    if example_kwarg_inputs is None:
        example_kwarg_inputs = {}
    return example_inputs, example_kwarg_inputs


def _process_export_inputs(mod, args, kwargs, dynamic_shapes):
    # Explicitly not calling mode.state_dict() as we do not want the module state for serialization
    # but the running module state so we can always match by id() the entries here with the graph inputs
    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    original_state_dict = named_parameters | named_buffers

    non_persistent_buffers = _get_non_persistent_buffers(mod)
    for k in non_persistent_buffers:
        original_state_dict.pop(k, None)

    if not isinstance(args, tuple):
        raise UserError(
            UserErrorType.INVALID_INPUT,
            f"Expecting `args` to be a tuple of example positional inputs, got {type(args)}",
        )
    kwargs = kwargs if kwargs is not None else {}
    _, original_in_spec = pytree.tree_flatten((args, kwargs))

    if isinstance(dynamic_shapes, torch.export.ShapesCollection):
        dynamic_shapes = dynamic_shapes.dynamic_shapes(mod, args, kwargs)

    return args, kwargs, original_in_spec, original_state_dict, dynamic_shapes


def _get_module_call_graph(
    export_artifact: ExportArtifact,
    original_in_spec: TreeSpec,
    preserve_module_call_signature: Tuple[str, ...],
    strict_mode_export: bool,
    forward_arg_names: Optional[List[str]] = None,
) -> Tuple[torch.fx.GraphModule, List[ModuleCallEntry]]:
    """
    In-place modify the graph module in export_artifact, remove _export_tracepoint nodes and
    return module_call_graph.
    """
    gm: torch.fx.GraphModule = export_artifact.aten.gm
    export_graph_signature: ExportGraphSignature = export_artifact.aten.sig
    module_call_specs: Dict[
        str, Dict[str, TreeSpec]
    ] = export_artifact.module_call_specs
    out_spec: TreeSpec = export_artifact.out_spec

    # Make module signatures.
    module_call_signatures: Dict[str, ModuleCallSignature] = {}
    for fqn, specs in module_call_specs.items():
        mod_fqn = _strip_root(fqn) if not strict_mode_export else fqn
        module_call_signatures[mod_fqn] = ModuleCallSignature(
            inputs=[],
            outputs=[],
            in_spec=specs["in_spec"],
            out_spec=specs["out_spec"],
            forward_arg_names=None,  # we only propage forward_arg_names for the top level module
        )

    if len(preserve_module_call_signature) > 0:
        if not strict_mode_export:
            _rewrite_tracepoint_node(gm)
        res = CollectTracepointsPass(module_call_signatures, export_graph_signature)(gm)
        assert res is not None
        gm = res.graph_module

    assert _EXPORT_MODULE_HIERARCHY is not None
    module_call_graph = _make_module_call_graph(
        original_in_spec,
        out_spec,
        module_call_signatures,
        forward_arg_names,
    )
    return gm, module_call_graph


def _get_range_constraints(
    export_artifact: ExportArtifact, combined_args: Dict[str, Any], dynamic_shapes
):
    gm: torch.fx.GraphModule = export_artifact.aten.gm
    export_graph_signature: ExportGraphSignature = export_artifact.aten.sig
    fake_mode: FakeTensorMode = export_artifact.fake_mode
    num_lifted = next(
        (
            i
            for i, s in enumerate(export_graph_signature.input_specs)
            if s.kind == InputKind.USER_INPUT
        ),
        len(export_graph_signature.input_specs),
    )
    range_constraints = make_constraints(
        fake_mode,
        gm,
        combined_args,
        dynamic_shapes,
        num_lifted,
    )
    return range_constraints


def _get_inline_constraints(fake_mode: FakeTensorMode):
    assert fake_mode.shape_env is not None
    return {
        k: v
        for k, v in fake_mode.shape_env.var_to_range.items()
        if free_unbacked_symbols(k)
    }


@contextmanager
def patch_forward(obj: torch.nn.Module, new_method):
    """Helper method to make it easier to cleanly torch.export() a method on a
    module that is not `forward`.
    """
    # Save the original method
    original_method = obj.forward

    # Patch the method
    obj.forward = new_method.__get__(obj, obj.__class__)

    try:
        yield
    finally:
        # Restore the original method
        obj.forward = original_method


@contextmanager
def _temp_disable_texpr_fuser():
    original_state = torch._C._jit_texpr_fuser_enabled()
    torch._C._jit_set_texpr_fuser_enabled(False)
    try:
        yield
    finally:
        torch._C._jit_set_texpr_fuser_enabled(original_state)


class _WrapperModule(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)


def _convert_ts_to_export_experimental(traced_callable, args, kwargs=None):
    with _temp_disable_texpr_fuser():
        from torch.jit._trace import TopLevelTracedModule

        export_args, export_kwargs = _process_jit_trace_inputs_for_export(args, kwargs)

        if isinstance(traced_callable, (TopLevelTracedModule, torch._C.ScriptModule)):  # type: ignore[operator]
            return _export(
                traced_callable,
                export_args,
                export_kwargs,
                strict=False,
                _is_torch_jit_trace=True,
            ).module()

        elif isinstance(traced_callable, torch.ScriptMethod) and isinstance(
            traced_callable.owner(), (torch._C.ScriptModule, torch.nn.Module)  # type: ignore[operator]
        ):
            with patch_forward(traced_callable.owner(), traced_callable):  # type: ignore[operator]
                return _export(
                    traced_callable.owner(),  # type: ignore[operator]
                    export_args,
                    export_kwargs,
                    strict=False,
                    _is_torch_jit_trace=True,
                ).module()

        else:
            return _export(
                _WrapperModule(traced_callable),
                export_args,
                export_kwargs,
                strict=False,
                _is_torch_jit_trace=True,
            ).module()


def _strict_export(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]],
    preserve_module_call_signature: Tuple[str, ...],
    pre_dispatch: bool,
    original_state_dict: Dict[str, Any],
    orig_in_spec: TreeSpec,
    allow_complex_guards_as_runtime_asserts: bool,
    _is_torch_jit_trace: bool,
) -> ExportArtifact:
    lower_to_aten = functools.partial(_export_to_aten_ir, pre_dispatch=pre_dispatch)
    return _strict_export_lower_to_aten_ir(
        mod=mod,
        args=args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        preserve_module_call_signature=preserve_module_call_signature,
        pre_dispatch=pre_dispatch,
        original_state_dict=original_state_dict,
        orig_in_spec=orig_in_spec,
        allow_complex_guards_as_runtime_asserts=allow_complex_guards_as_runtime_asserts,
        _is_torch_jit_trace=_is_torch_jit_trace,
        lower_to_aten_callback=lower_to_aten,
    )


def _strict_export_lower_to_aten_ir(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]],
    preserve_module_call_signature: Tuple[str, ...],
    pre_dispatch: bool,
    original_state_dict: Dict[str, Any],
    orig_in_spec: TreeSpec,
    allow_complex_guards_as_runtime_asserts: bool,
    _is_torch_jit_trace: bool,
    lower_to_aten_callback: Callable,
) -> ExportArtifact:
    gm_torch_level = _export_to_torch_ir(
        mod,
        args,
        kwargs,
        dynamic_shapes,
        preserve_module_call_signature=preserve_module_call_signature,
        restore_fqn=False,  # don't need to restore because we will do it later
        allow_complex_guards_as_runtime_asserts=allow_complex_guards_as_runtime_asserts,
        _log_export_usage=False,
    )

    # We detect the fake_mode by looking at gm_torch_level's placeholders, this is the fake_mode created in dynamo.
    (
        fake_args,
        fake_kwargs,
        dynamo_fake_mode,
    ) = _extract_fake_inputs(gm_torch_level, args, kwargs)

    fake_params_buffers = _fakify_params_buffers(dynamo_fake_mode, gm_torch_level)

    # First, we want to pass through the graph to try populating
    # val field for getattr if there is anything missing.
    # This can happen when quantization adds extra params and forgets
    # to update "val"
    for node in gm_torch_level.graph.nodes:
        if node.op == "get_attr" and "val" not in node.meta:
            attr = getattr(gm_torch_level, node.target)
            # Checks if it is not a HigherOrderOp branch or a module
            if not isinstance(attr, torch.nn.Module):
                assert (
                    dynamo_fake_mode is not None
                ), "Cannot find dynamo_fake_mode. This could be due to the exported graph module have no placeholders."
                node.meta["val"] = dynamo_fake_mode.from_tensor(
                    attr, static_shapes=True
                )

    # Fix the graph output signature to be tuple if scalar

    # gm_torch_level.graph._codegen is made a _PyTreeCodeGen in rewrite_signature in eval_frame.py
    assert isinstance(gm_torch_level.graph._codegen, torch.fx.graph._PyTreeCodeGen)

    # Calling gm_torch_level._out_spec is not safe because gm_torch_level might be
    # a _LazyGraphModule, which does not populate _out_spec when calling recompile().
    # TODO: Fix recompile() in  _LazyGraphModule. T207713214
    out_spec = orig_out_spec = gm_torch_level.graph._codegen.pytree_info.out_spec

    # Used to get rid of lint type error.
    assert out_spec is not None
    assert orig_out_spec is not None

    # aot_export expect the return type to always be a tuple.
    if out_spec.type not in (list, tuple):
        out_spec = pytree.TreeSpec(tuple, None, [out_spec])

    orig_arg_names = gm_torch_level.graph._codegen.pytree_info.orig_args  # type: ignore[attr-defined]

    gm_torch_level.graph._codegen = _PyTreeCodeGen(
        _PyTreeInfo(
            orig_arg_names,
            gm_torch_level._in_spec,
            out_spec,
        )
    )
    gm_torch_level.recompile()

    _normalize_nn_module_stack(gm_torch_level, type(mod))

    params_buffers_to_node_meta = _collect_param_buffer_metadata(gm_torch_level)

    # When aot_export lifts the params, we lose metadata (e.g. source_fn_stack, stack_trace)
    # from the param nodes as they are treated as fresh inputs
    # Therefore, we manually extract them before calling into aot_export
    # params_buffers_to_node_meta = _collect_param_buffer_metadata(gm_torch_level)

    constant_attrs = _gather_constant_attrs(mod)
    param_buffer_table: Dict[str, str] = _get_param_buffer_mapping(mod, gm_torch_level)

    # Dynamo does not track which buffers were registered as non-persistent. This info
    # is available in the original module, so we transfer it to the traced module. Also,
    # since we didn't restore original param/buffer names yet, we must use traced names.
    non_persistent_buffers = _get_non_persistent_buffers(mod)
    reverse_name_lookup = {orig: traced for traced, orig in param_buffer_table.items()}
    gm_torch_level._non_persistent_buffers_set = {
        reverse_name_lookup[name]
        for name in non_persistent_buffers
        if name in reverse_name_lookup
    }
    with dynamo_fake_mode:
        aten_export_artifact = lower_to_aten_callback(
            gm_torch_level,
            # NOTE: graph module expects only positional args
            _convert_to_positional_args(orig_arg_names, fake_args, fake_kwargs),
            {},
            fake_params_buffers,
            constant_attrs,
        )

    # Decompose for readability.
    gm = aten_export_artifact.gm
    export_graph_signature = aten_export_artifact.sig
    constants = aten_export_artifact.constants

    _populate_param_buffer_metadata_to_new_gm(
        params_buffers_to_node_meta, gm, export_graph_signature
    )

    # Do some cleanups on the graph module to restore the state dict to the
    # expected form. Each of these steps should probably get fixed upstream.
    # 1. Remove tensor constants that were added as buffers.
    _rewrite_dynamo_tensor_constants(
        orig_mod_buffers=set(mod.buffers()),
        traced_mod_buffers=dict(gm_torch_level.named_buffers()),
        graph_signature=export_graph_signature,
        constants=constants,
    )
    # 2. Restore FQN of param/buffers
    _replace_param_buffer_names(param_buffer_table, export_graph_signature)

    # 3. Move non-persistent buffers to tensor constants
    _move_non_persistent_buffers_to_tensor_constants(
        mod, export_graph_signature, constants
    )

    # 4. Rewrite constants to have the same FQN as the original module.
    _remap_constants(constant_attrs, export_graph_signature, constants)

    # 5. Rename constants nodes in graph module from buffers to constants
    _rename_constants_nodes(gm, export_graph_signature)

    return ExportArtifact(
        aten=aten_export_artifact,
        out_spec=orig_out_spec,
        fake_mode=dynamo_fake_mode,
        module_call_specs=gm_torch_level.meta["module_call_specs"],
    )


def _export_to_aten_ir_make_fx(
    mod: torch.nn.Module,
    fake_args,
    fake_kwargs,
    fake_params_buffers,
    constant_attrs: ConstantAttrMap,
    produce_guards_callback=None,
    transform=lambda x: x,
) -> ATenExportArtifact:
    def _make_fx_helper(mod, args, kwargs, **flags):
        from torch._functorch._aot_autograd.schemas import GraphSignature

        kwargs = kwargs or {}

        named_parameters = dict(mod.named_parameters(remove_duplicate=False))
        named_buffers = dict(mod.named_buffers(remove_duplicate=False))

        params_and_buffers = {**named_parameters, **named_buffers}
        params_and_buffers_flat, params_spec = pytree.tree_flatten(params_and_buffers)
        params_and_buffers_flat = tuple(params_and_buffers_flat)

        param_len = len(named_parameters)
        buffer_len = len(named_buffers)
        params_len = len(params_and_buffers)

        functional_call = create_functional_call(
            mod, params_spec, params_len, store_orig_mod=True
        )

        params_buffers_args: List[Any] = []
        params_buffers_args.extend(params_and_buffers_flat)
        params_buffers_args.extend(args)

        flat_fn, out_spec = create_tree_flattened_fn(
            functional_call, params_buffers_args, kwargs
        )
        flat_args, in_spec = pytree.tree_flatten((params_buffers_args, kwargs))

        @functools.wraps(flat_fn)
        def wrapped_fn(*args):
            return tuple(flat_fn(*args))

        with enable_python_dispatcher():
            ctx = nullcontext()
            non_strict_root = getattr(mod, "_export_root", None)
            if non_strict_root is not None:
                ctx = _detect_attribute_assignment(non_strict_root)  # type: ignore[assignment]

                # For any buffer that is assigned, we want to associate it to the final proxy node
                # that it is assigned to. This node can then be copied into the buffer.
                assigned_buffers: Dict[str, str] = {}
                hook = register_buffer_assignment_hook(
                    non_strict_root, assigned_buffers
                )

            with ctx:
                gm = make_fx(
                    wrapped_fn,
                    record_module_stack=True,
                    pre_dispatch=True,
                )(*flat_args)

            if non_strict_root is not None:
                input_names = _graph_input_names(gm)
                buffer_input_names = {
                    buf: input_names[param_len + i]
                    for i, buf in enumerate(non_strict_root._buffers)
                }
                output_node = list(gm.graph.nodes)[-1]
                # We copy nodes corresponding to buffer assignments to buffers in the graph.
                for buf, name in assigned_buffers.items():  # type: ignore[possibly-undefined]
                    buf_node = _find_node(gm, buffer_input_names[buf])
                    name_node = _find_node(gm, name)
                    with gm.graph.inserting_before(output_node):
                        new_node = gm.graph.create_node(
                            "call_function",
                            torch.ops.aten.copy_.default,
                            args=(buf_node, name_node),
                        )
                        new_node.meta = name_node.meta

                hook.remove()  # type: ignore[possibly-undefined]

            # In export, we ignore any op that is related to
            # eager mode profiling call. The expectation is
            # that either runtimes provide their own profiling
            # OR user wrap the compiled region on a profiling in
            # later stage.
            def _is_impure(node):
                if node.op == "call_function" and node.target in (
                    torch.ops.profiler._record_function_enter.default,
                    torch.ops.profiler._record_function_enter_new.default,
                    torch.ops.profiler._record_function_exit._RecordFunction,
                ):
                    return False
                return True

            gm.graph.eliminate_dead_code(_is_impure)

        # create graph signature
        input_names = _graph_input_names(gm)
        output_names = _graph_output_names(gm)
        sig = GraphSignature(
            parameters=list(named_parameters),
            buffers=list(named_buffers),
            user_inputs=input_names[params_len:],
            user_outputs=output_names,
            inputs_to_parameters=dict(zip(input_names[0:param_len], named_parameters)),
            inputs_to_buffers=dict(
                zip(input_names[param_len : param_len + buffer_len], named_buffers)
            ),
            buffers_to_mutate={},
            user_inputs_to_mutate={},
            in_spec=in_spec,
            out_spec=out_spec,  # type: ignore[arg-type]
            backward_signature=None,
            input_tokens=[],
            output_tokens=[],
        )
        return gm, sig

    # This _reparametrize_module makes sure inputs and module.params/buffers have the same fake_mode,
    # otherwise aot_export_module will error out because it sees a mix of fake_modes.
    # And we want aot_export_module to use the fake_tensor mode in dynamo to keep the pipeline easy to reason about.
    with torch.nn.utils.stateless._reparametrize_module(
        mod,
        fake_params_buffers,
        tie_weights=True,
        strict=True,
        stack_weights=True,
    ), _ignore_backend_decomps(), _compiling_state_context():  # type: ignore[attr-defined]
        param_len = len(dict(mod.named_parameters(remove_duplicate=False)))
        buffer_len = len(dict(mod.named_buffers(remove_duplicate=False)))
        params_len = param_len + buffer_len

        gm, graph_signature = transform(_make_fx_helper)(
            mod,
            fake_args,
            trace_joint=False,
            kwargs=fake_kwargs,
        )

        # [NOTE] In training IR, we don't run
        # any DCE as a result we preserve constant
        # nodes in the graph. make_fx invariant is that
        # they don't guarantee every node gets a meta['val']
        # field. Since the actual value is already hardcoded in
        # graph, the node.meta here actually doesn't matter. But
        # we do this to make spec verifier happy.
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and len(node.users) == 0
                and "val" not in node.meta
            ):
                node.meta["val"] = None

        if isinstance(mod, torch.fx.GraphModule) and hasattr(mod, "meta"):
            gm.meta.update(mod.meta)

    # See comment in _export_to_aten_ir()
    if produce_guards_callback:
        try:
            produce_guards_callback(gm)
        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e))  # noqa: B904

    return _produce_aten_artifact(
        gm=gm,
        mod=mod,
        constant_attrs=constant_attrs,
        graph_signature=graph_signature,
        pre_dispatch=True,
        fake_args=fake_args,
        fake_kwargs=fake_kwargs,
        fake_params_buffers=fake_params_buffers,
    )


def set_missing_meta_vals(gm, flat_args, num_params_buffers):
    # Sets missing metadata to address two problems:
    # 1. aot_export adds symint metadata for placeholders with int values; since
    #    these become specialized, we replace such metadata with the original values.
    # 2. any tensor attributes that are not params / buffers, i.e., are constants
    #    need to have their metadata set before lifting them because it is needed
    #    for computing the exported program's signature.
    index = 0
    fake_mode = detect_fake_mode(flat_args)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if index >= num_params_buffers:
                user_arg = flat_args[index - num_params_buffers]
                if not isinstance(user_arg, torch.Tensor):
                    node.meta["val"] = user_arg
            index += 1
        if node.op == "get_attr":
            val = _get_attr(gm, node.target)
            if isinstance(val, torch.Tensor):
                assert "val" not in node.meta, (
                    f"Found attribute {node.target} that has already been fakified "
                    "but not yet lifted as an input. This should be impossible because "
                    "(1) we should have already fakified AND lifted params/buffers "
                    "(2) we should have NOT yet fakified OR lifted tensor constants. "
                )
                node.meta["val"] = fake_mode.from_tensor(val, static_shapes=True)


def _find_node(gm: torch.fx.GraphModule, name: str) -> torch.fx.Node:
    return next(iter(node for node in gm.graph.nodes if node.name == name))


def _non_strict_export(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]],
    preserve_module_call_signature: Tuple[str, ...],
    pre_dispatch: bool,
    original_state_dict: Dict[str, Any],
    orig_in_spec: TreeSpec,
    allow_complex_guards_as_runtime_asserts: bool,
    _is_torch_jit_trace: bool,
    dispatch_tracing_mode: str = "aot_export",
) -> ExportArtifact:
    """
    ``dispatch_tracing_mode`` can be either "make_fx” or “aot_export”, corresponding to
    _export_to_aten_ir_make_fx and _export_to_aten_ir, respectively.
    """
    assert dispatch_tracing_mode in ["make_fx", "aot_export"]
    out_spec: Optional[TreeSpec] = None

    module_call_specs: Dict[str, Dict[str, pytree.TreeSpec]] = {}

    def _tuplify_outputs(aot_export):
        def _aot_export_non_strict(mod, args, kwargs=None, **flags):
            kwargs = kwargs or {}

            class Wrapper(torch.nn.Module):
                def __init__(self, mod):
                    super().__init__()
                    self._export_root = mod

                def forward(self, *args, **kwargs):
                    nonlocal out_spec
                    mod = self._export_root
                    if isinstance(mod, torch.fx.GraphModule):
                        # NOTE: We're going to run this graph module with an fx interpreter,
                        # which will not run any forward hooks. Thus, ideally, we should run
                        # all forward hooks here. But the general logic for running them is
                        # complicated (see nn/module.py), and probably not worth duplicating.
                        # Instead we only look for, and run, an export-specific forward hook.
                        if (
                            _check_input_constraints_pre_hook
                            in mod._forward_pre_hooks.values()
                        ):
                            _check_input_constraints_pre_hook(mod, args, kwargs)
                        with torch.fx.traceback.preserve_node_meta():
                            args = (*args, *kwargs.values())
                            tree_out = torch.fx.Interpreter(mod).run(*args)
                    else:
                        tree_out = mod(*args, **kwargs)
                    flat_outs, out_spec = pytree.tree_flatten(tree_out)
                    return tuple(flat_outs)

            wrapped_mod = Wrapper(mod)
            # Patch export_root to the signatures so that wrapper module correctly populates the
            # in/out spec
            new_preserved_call_signatures = [
                "_export_root." + i for i in preserve_module_call_signature
            ]
            with _wrap_submodules(
                wrapped_mod, new_preserved_call_signatures, module_call_specs
            ):
                gm, sig = aot_export(wrapped_mod, args, kwargs=kwargs, **flags)
                log.debug("Exported program from AOTAutograd:\n%s", gm)

            sig.parameters = pytree.tree_map(_strip_root, sig.parameters)
            sig.buffers = pytree.tree_map(_strip_root, sig.buffers)
            sig.inputs_to_buffers = pytree.tree_map(_strip_root, sig.inputs_to_buffers)
            sig.inputs_to_parameters = pytree.tree_map(
                _strip_root, sig.inputs_to_parameters
            )
            sig.buffers_to_mutate = pytree.tree_map(_strip_root, sig.buffers_to_mutate)

            for node in gm.graph.nodes:
                if "nn_module_stack" in node.meta:
                    nn_module_stack = node.meta["nn_module_stack"]
                    node.meta["nn_module_stack"] = {
                        _fixup_key(key): val
                        for key, val in pytree.tree_map(
                            _strip_root, nn_module_stack
                        ).items()
                    }

            return gm, sig

        return _aot_export_non_strict

    (
        fake_mode,
        fake_args,
        fake_kwargs,
        equalities_inputs,
        original_signature,
        dynamic_shapes,
    ) = make_fake_inputs(
        mod,
        args,
        kwargs,
        dynamic_shapes,
        _is_torch_jit_trace=_is_torch_jit_trace,
        allow_complex_guards_as_runtime_asserts=allow_complex_guards_as_runtime_asserts,  # for shape env initialization
    )

    fake_params_buffers = _fakify_params_buffers(fake_mode, mod)

    def _produce_guards_callback(gm):
        return produce_guards_and_solve_constraints(
            fake_mode=fake_mode,
            gm=gm,
            dynamic_shapes=dynamic_shapes,
            equalities_inputs=equalities_inputs,
            original_signature=original_signature,
            _is_torch_jit_trace=_is_torch_jit_trace,
        )

    with fake_mode, _NonStrictTorchFunctionHandler():
        with _fakify_script_objects(mod, fake_args, fake_kwargs, fake_mode) as (
            patched_mod,
            new_fake_args,
            new_fake_kwargs,
            new_fake_constant_attrs,
            map_fake_to_real,
        ):
            _to_aten_func = (
                _export_to_aten_ir_make_fx
                if dispatch_tracing_mode == "make_fx"
                else functools.partial(
                    _export_to_aten_ir,
                    pre_dispatch=pre_dispatch,
                    _is_torch_jit_trace=_is_torch_jit_trace,
                )
            )
            aten_export_artifact = _to_aten_func(  # type: ignore[operator]
                patched_mod,
                new_fake_args,
                new_fake_kwargs,
                fake_params_buffers,
                new_fake_constant_attrs,
                produce_guards_callback=_produce_guards_callback,
                transform=_tuplify_outputs,
            )
            # aten_export_artifact.constants contains only fake script objects, we need to map them back
            aten_export_artifact.constants = {
                fqn: map_fake_to_real[obj] if isinstance(obj, FakeScriptObject) else obj
                for fqn, obj in aten_export_artifact.constants.items()
            }

    _move_non_persistent_buffers_to_tensor_constants(
        mod, aten_export_artifact.sig, aten_export_artifact.constants
    )

    assert out_spec is not None

    return ExportArtifact(
        aten=aten_export_artifact,
        out_spec=out_spec,
        fake_mode=fake_mode,
        module_call_specs=module_call_specs,
    )


@_log_export_wrapper
@_disable_prexisiting_fake_mode
def _export_for_training(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]] = None,
    *,
    strict: bool = True,
    preserve_module_call_signature: Tuple[str, ...] = (),
) -> ExportedProgram:
    global _EXPORT_MODULE_HIERARCHY
    _EXPORT_MODULE_HIERARCHY = _get_module_hierarchy(mod)

    (
        args,
        kwargs,
        orig_in_spec,
        original_state_dict,
        dynamic_shapes,
    ) = _process_export_inputs(mod, args, kwargs, dynamic_shapes)

    export_func = (
        functools.partial(
            _strict_export_lower_to_aten_ir,
            lower_to_aten_callback=_export_to_aten_ir_make_fx,
        )
        if strict
        else functools.partial(
            _non_strict_export,
            dispatch_tracing_mode="make_fx",
        )
    )
    export_artifact = export_func(  # type: ignore[operator]
        mod=mod,
        args=args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        preserve_module_call_signature=preserve_module_call_signature,
        pre_dispatch=False,
        original_state_dict=original_state_dict,
        orig_in_spec=orig_in_spec,
        allow_complex_guards_as_runtime_asserts=False,
        _is_torch_jit_trace=False,
    )

    export_graph_signature = export_artifact.aten.sig

    forward_arg_names = _get_forward_arg_names(mod, args, kwargs)
    inline_constraints = _get_inline_constraints(export_artifact.fake_mode)
    # The unbacked symint symbols are updated in aot_export
    # so we serialize them here instead of inside dynamo.
    # Note: _get_range_constraints depends on "inline_constraints" to be set.
    export_artifact.aten.gm.meta["inline_constraints"] = inline_constraints
    range_constraints = _get_range_constraints(
        export_artifact,
        _combine_args(mod, args, kwargs, _is_torch_jit_trace=False),
        dynamic_shapes,
    )
    # The returned the gm is in-place modified
    gm, module_call_graph = _get_module_call_graph(
        export_artifact,
        orig_in_spec,
        preserve_module_call_signature,
        strict,
        forward_arg_names,
    )

    _verify_nn_module_stack(gm)
    _verify_stack_trace(gm)
    _verify_placeholder_names(gm, export_graph_signature)

    _update_gm_meta_if_possible(gm, mod)

    from torch._export.verifier import TrainingIRVerifier

    exported_program = ExportedProgram(
        root=gm,
        graph=gm.graph,
        graph_signature=export_graph_signature,
        state_dict=original_state_dict,
        range_constraints=range_constraints,
        module_call_graph=module_call_graph,
        example_inputs=(args, kwargs),
        constants=export_artifact.aten.constants,
        verifiers=[TrainingIRVerifier],
    )

    return exported_program


@_log_export_wrapper
@_disable_prexisiting_fake_mode
def _export(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]] = None,
    *,
    strict: bool = True,
    preserve_module_call_signature: Tuple[str, ...] = (),
    pre_dispatch: bool = False,
    allow_complex_guards_as_runtime_asserts: bool = False,
    _is_torch_jit_trace: bool = False,
) -> ExportedProgram:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a ExportedProgram.

    Args:
        f: the `nn.Module` to trace.

        args: example positional inputs.

        kwargs: optional example keyword inputs.

        dynamic_shapes:
         An optional argument where the type should either be:
         1) a dict from argument names of ``f`` to their dynamic shape specifications,
         2) a tuple that specifies dynamic shape specifications for each input in original order.
         If you are specifying dynamism on keyword args, you will need to pass them in the order that
         is defined in the original function signature.

         The dynamic shape of a tensor argument can be specified as either
         (1) a dict from dynamic dimension indices to :func:`Dim` types, where it is
         not required to include static dimension indices in this dict, but when they are,
         they should be mapped to None; or (2) a tuple / list of :func:`Dim` types or None,
         where the :func:`Dim` types correspond to dynamic dimensions, and static dimensions
         are denoted by None. Arguments that are dicts or tuples / lists of tensors are
         recursively specified by using mappings or sequences of contained specifications.

        preserve_module_call_signature: A list of submodule paths for which the original
            calling conventions are preserved as metadata.

        allow_complex_guards_as_runtime_asserts:
         With the current dynamic shapes language for dims and derived dims, we can run into constraints
         that are not expressible with the language. For example, flattening a matrix and adding to a vector,
         both fully dynamic (i.e. x.reshape([-1]) + y) emits a guard s0 * s1 = s2, which is not expressible.
         By default, we either raise a constraint violation error or specialize to static values.
         If this flag is set to True, we avoid erroring out and instead allow complex constraints to exist as runtime
         assertions in the graph. The sympy interpreter (torch/utils/_sympy/interp.py) will produce the math ops
         required to compute and assert the value of the guard (e.g. sym_size_int, eq, _assert_scalar).
         Additionally, if TORCH_DYNAMO_DO_NOT_EMIT_RUNTIME_ASSERTS=1 is specified, we will allow complex constraints
         while not emitting runtime asserts, returning a cleaner graph with lesser guarantees around dynamic shapes.

    Returns:
        An ExportedProgram containing the traced method.
    """

    from torch._utils_internal import export_training_ir_rollout_check

    global _EXPORT_FLAGS, _EXPORT_MODULE_HIERARCHY
    _EXPORT_MODULE_HIERARCHY = _get_module_hierarchy(mod)

    flags = set()
    flags.add("strict" if strict else "non_strict")
    flags.add("pre_dispatch" if pre_dispatch else "aot_dispatch")
    _EXPORT_FLAGS = flags

    log_export_usage(event="export.enter", flags=_EXPORT_FLAGS)

    # NOTE Export training IR rollout
    # Old export calls export._trace(pre_dispatch=True)
    # and there are still lot of internal/OSS callsites that
    # use export._trace(pre_dispatch=True) directly. Therefore,
    # it makes more sense to do the switch here.
    # export_training_ir_rollout_check returns True in OSS
    # while internally it returns False UNLESS otherwise specified.
    if pre_dispatch and export_training_ir_rollout_check():
        return _export_for_training(
            mod,
            args,
            kwargs,
            dynamic_shapes,
            strict=strict,
            preserve_module_call_signature=preserve_module_call_signature,
        )

    (
        args,
        kwargs,
        original_in_spec,
        original_state_dict,
        dynamic_shapes,
    ) = _process_export_inputs(mod, args, kwargs, dynamic_shapes)

    # Call the appropriate export function based on the strictness of tracing.
    export_func = _strict_export if strict else _non_strict_export

    export_artifact = export_func(  # type: ignore[operator]
        mod,
        args,
        kwargs,
        dynamic_shapes,
        preserve_module_call_signature,
        pre_dispatch,
        original_state_dict,
        original_in_spec,
        allow_complex_guards_as_runtime_asserts,
        _is_torch_jit_trace,
    )
    export_graph_signature: ExportGraphSignature = export_artifact.aten.sig

    forward_arg_names = (
        _get_forward_arg_names(mod, args, kwargs) if not _is_torch_jit_trace else None
    )
    inline_constraints = _get_inline_constraints(export_artifact.fake_mode)
    # The unbacked symint symbols are updated in aot_export
    # so we serialize them here instead of inside dynamo.
    # Note: this step must be before _get_range_constraints.
    export_artifact.aten.gm.meta["inline_constraints"] = inline_constraints
    range_constraints = _get_range_constraints(
        export_artifact,
        _combine_args(mod, args, kwargs, _is_torch_jit_trace=_is_torch_jit_trace),
        dynamic_shapes,
    )
    gm, module_call_graph = _get_module_call_graph(
        export_artifact,
        original_in_spec,
        preserve_module_call_signature,
        strict,
        forward_arg_names,
    )

    _verify_nn_module_stack(gm)
    _verify_stack_trace(gm)
    if not _is_torch_jit_trace:
        _verify_placeholder_names(gm, export_graph_signature)

    # Remove Proxy because they cannot be deepcopied or pickled.
    torch._export.utils.remove_proxy_from_state_dict(original_state_dict, in_place=True)

    from torch._export.verifier import Verifier

    _update_gm_meta_if_possible(gm, mod)

    exported_program = ExportedProgram(
        root=gm,
        graph=gm.graph,
        graph_signature=export_graph_signature,
        state_dict=original_state_dict,
        range_constraints=range_constraints,
        module_call_graph=module_call_graph,
        example_inputs=(args, kwargs),
        constants=export_artifact.aten.constants,
        verifiers=[Verifier],
    )

    return exported_program
