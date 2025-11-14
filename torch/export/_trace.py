# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import dataclasses
import functools
import inspect
import logging
import re
import sys
import time
import warnings
from collections.abc import Callable
from contextlib import contextmanager, ExitStack, nullcontext
from itertools import chain
from typing import Any, Optional, TYPE_CHECKING, TypeAlias, Union
from unittest import mock


if TYPE_CHECKING:
    import weakref

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
    _fakify_module_inputs,
    _fakify_script_objects,
    _gather_constant_attrs,
    _NonStrictTorchFunctionHandler,
    _override_builtin_ops,
    make_constraints,
    make_fake_inputs,
    produce_guards_and_solve_constraints,
)
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass
from torch._export.passes.lift_constants_pass import (
    _materialize_and_lift_constants,
    ConstantAttrMap,
)
from torch._export.utils import (
    _collect_param_buffer_metadata,
    _compiling_state_context,
    _fakify_params_buffers,
    _populate_param_buffer_metadata_to_new_gm,
    _update_gm_meta_if_possible,
    apply_runtime_assertion_pass,
    placeholder_naming_pass,
    placeholder_prefixes,
)
from torch._export.verifier import SpecViolationError
from torch._export.wrappers import _wrap_submodules
from torch._functorch._aot_autograd.graph_capture_wrappers import create_functional_call
from torch._functorch._aot_autograd.input_output_analysis import (
    _graph_input_names,
    _graph_output_names,
)
from torch._functorch._aot_autograd.schemas import GraphSignature
from torch._functorch._aot_autograd.subclass_utils import get_subclass_typing_container
from torch._functorch._aot_autograd.utils import (
    create_tree_flattened_fn,
    register_buffer_assignment_hook,
)
from torch._functorch.aot_autograd import (
    _detect_attribute_assignment,
    aot_export_joint_with_descriptors,
)
from torch._guards import detect_fake_mode, tracing, TracingContext
from torch._library.fake_class_registry import FakeScriptObject
from torch._logging import dtrace_structured
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._utils_internal import log_export_usage
from torch.export._leakage_detection_utils import find_legit_leaks_from_referrers
from torch.export._unlift import _check_input_constraints_pre_hook
from torch.export.dynamic_shapes import (
    _check_dynamic_shapes,
    _combine_args,
    _DimHintType,
    _IntWrapper,
    _process_dynamic_shapes,
)
from torch.export.exported_program import OutputKind
from torch.fx._symbolic_trace import _ConstantAttributeType
from torch.fx.experimental.proxy_tensor import (
    get_proxy_slot,
    make_fx,
    PreDispatchTorchFunctionMode,
    track_tensor_tree,
)
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
    ShapeEnv,
)
from torch.fx.graph import _PyTreeInfo
from torch.utils._pytree import TreeSpec
from torch.utils._sympy.value_ranges import ValueRangeError

from .exported_program import (
    _disable_prexisiting_fake_mode,
    ExportedProgram,
    InputKind,
    ModuleCallEntry,
    ModuleCallSignature,
)
from .graph_signature import _convert_to_export_graph_signature, ExportGraphSignature


log = logging.getLogger(__name__)

# Type alias for dynamic shapes specification
_DynamicShapesSpec: TypeAlias = Union[dict[str, Any], tuple[Any, ...], list[Any]]


@dataclasses.dataclass
class ExportDynamoConfig:
    """
    Manage Export-specific configurations of Dynamo.
    """

    allow_rnn: bool = True
    reorderable_logging_functions: set[Callable] = dataclasses.field(
        default_factory=set
    )
    # Emit runtime asserts after AOTAutograd instead.
    # This isn't really necessary, and isn't much more efficient since the runtime asserts pass does CSE,
    # but if we want to reason more about what guards/runtime asserts to emit,
    # this makes it a bit cleaner to do from the export side. Also no real point in running this twice.
    do_not_emit_runtime_asserts: bool = True
    specialize_int: bool = True
    specialize_float: bool = True
    assume_static_by_default: bool = False
    automatic_dynamic_shapes: bool = False
    capture_dynamic_output_shape_ops: bool = True
    capture_scalar_outputs: bool = True
    prefer_deferred_runtime_asserts_over_guards: bool = False
    replay_side_effects: bool = False
    side_effect_replay_policy: str = "warn"


@dataclasses.dataclass
class ATenExportArtifact:
    gm: torch.fx.GraphModule
    sig: ExportGraphSignature
    constants: dict[str, _ConstantAttributeType]


@dataclasses.dataclass(frozen=True)
class ExportArtifact:
    aten: ATenExportArtifact
    in_spec: TreeSpec
    out_spec: TreeSpec
    fake_mode: FakeTensorMode
    module_call_specs: dict[str, dict[str, pytree.TreeSpec]]


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


@contextmanager
def _disable_custom_triton_op_functional_decomposition():
    old = torch._functorch.config.decompose_custom_triton_ops
    try:
        # pyrefly: ignore [bad-assignment]
        torch._functorch.config.decompose_custom_triton_ops = False
        yield torch._functorch.config.decompose_custom_triton_ops
    finally:
        torch._functorch.config.decompose_custom_triton_ops = old


def custom_triton_ops_decomposition_disabled():
    return not torch._functorch.config.decompose_custom_triton_ops


def _fixup_key(x):
    return "L__self__" + _strip_root(x)


def _strip_root(x):
    if isinstance(x, str) and x.startswith("_export_root"):
        stripped = x[len("_export_root") :]
        return stripped.removeprefix(".")
    return x


def _is_bogus_const_name(name: str):
    splitted_names = name.split(".")
    if len(splitted_names) < 1:
        return True

    return splitted_names[-1].startswith("lifted_tensor")


def _rewrite_tracepoint_node(gm: torch.fx.GraphModule):
    """
    In-place modify input graph module by replacing the export tracepoint with a new node
    that has the same target and args, but with the _export_root stripped from path.
    """
    for node in gm.graph.nodes:
        if node.target is torch.ops.higher_order._export_tracepoint:
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


def detect_shape_env(inputs: Any = None):
    shape_envs = []

    for i, flat_input in enumerate(inputs):
        if isinstance(flat_input, torch.SymInt):
            shape_envs.append((flat_input.node.shape_env, "symint input", i))

    if shape_envs:
        shape_env, desc1, i1 = shape_envs[0]
        for m, desc2, i2 in shape_envs[1:]:
            assert shape_env is m, (
                f"shape env ({shape_env}) from {desc1} {i1} doesn't match mode ({m}) from {desc2} {i2}\n\n"
                f"shape env from {desc1} {i1} allocated at:\n{shape_env.stack}\n"
                f"shape env from {desc2} {i2} allocated at:\n{m.stack}"
            )
        return shape_env
    else:
        return None


def _extract_fake_inputs(gm, args, kwargs):
    """
    Given a graph module, extract fakified input tensors from the metadata of
    its placeholders, and map them to the structure of given args and kwargs.
    Also return the fake mode used to fakify those inputs.
    """
    fake_inps: list[Any] = []
    fake_vals: list[Any] = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            fake_inps.append(node.meta.get("val"))
        else:
            fake_vals.append(node.meta.get("example_value"))

    if in_shuffle_graph := getattr(gm, "_in_shuffle_graph", None):
        flat_args = pytree.tree_leaves((args, kwargs))
        node_map = {
            node: i
            for i, node in enumerate(
                next(iter(reversed(in_shuffle_graph.graph.nodes))).args[0]
            )
            if node.op == "placeholder"
        }
        new_fake_inps: list[Any] = []
        for i, node in enumerate(
            in_shuffle_graph.graph.find_nodes(op="placeholder")[1:]
        ):
            if node in node_map:
                new_fake_inps.append(fake_inps[node_map[node]])
            else:
                new_fake_inps.append(flat_args[i])
        fake_inps = new_fake_inps
    # We get both because now we might have a combination of symint and tensor
    # inputs, and we want to check that the shape env is consistent between
    # both. Unfortunately we can't see what fake mode is attached to the shape
    # env, then we can just compare fake modes.
    detected_fake_mode = detect_fake_mode(fake_inps + fake_vals)
    detected_shape_env = detect_shape_env(fake_inps + fake_vals)

    if detected_fake_mode:
        if detected_shape_env:
            assert detected_shape_env is detected_fake_mode.shape_env, (
                "Detected shape env does not match fake mode's shape env"
            )
        fake_mode = detected_fake_mode
    elif detected_shape_env:
        fake_mode = FakeTensorMode(shape_env=detected_shape_env, export=True)
    else:
        fake_mode = FakeTensorMode(shape_env=ShapeEnv(), export=True)

    count = 0

    def lookup_fake(x):
        nonlocal count
        val = fake_inps[count] if isinstance(x, (int, torch.Tensor)) else x
        count += 1
        return val

    fake_args = pytree.tree_map(lookup_fake, args)
    fake_kwargs = pytree.tree_map(lookup_fake, kwargs)

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
                    if path == "L['self']":
                        return ""
                    if path.startswith("L['self']."):
                        return path[len("L['self'].") :]
                    return path

                nn_module_stack = {
                    root_key: (root, root_cls.__module__ + "." + root_cls.__qualname__),
                    # pyrefly: ignore [unbound-name]
                    **nn_module_stack,
                }
                node.meta["nn_module_stack"] = {
                    key: (normalize_path(path), ty)
                    for key, (path, ty) in nn_module_stack.items()
                }


def _get_param_buffer_mapping(
    original_module: torch.nn.Module,
    traced_module: torch.nn.Module,
) -> dict[str, str]:
    """
    Returns a mapping of parameter/buffer names from the new module to the
    original model. This is to help with restoring the FQN for parameter/buffers
    of a traced module to what the original module contains.
    """

    param_lookup: dict[int, str] = {}
    buffer_lookup: dict[int, str] = {}
    for name, param in original_module.named_parameters(remove_duplicate=False):
        if param_lookup.get(id(param)) is None:
            # we only want to keep the first occurrence of a parameter to guarantee parity of original and traced module.
            param_lookup[id(param)] = name
    for name, buffer in original_module.named_buffers(remove_duplicate=False):
        buffer_lookup[id(buffer)] = name

    param_buffer_table: dict[str, str] = {}
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
    fake_params_buffers: dict[str, torch.Tensor],
    constants: dict[str, _ConstantAttributeType],
    flat_fake_args: list[Any],
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
    constants: dict[str, _ConstantAttributeType],
) -> None:
    """Rewrite the graph signature and constants table to use the FQN from the original module."""
    remap_table: dict[str, list[str]] = {}
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


def _replace_unbacked_bindings(gm: torch.fx.GraphModule) -> None:
    """
    When we run an interpreter-based pass over a GraphModule, execution of data-dependent operators
    will produce example values with new unbacked symbols. To track that the new/old symbols are equivalent,
    we used to rely on the unbacked_renamings mapping. This led to problematic metadata where the unbacked_bindings
    keys mapped new symbols (u2) to paths containing old symbols (u0) in the example values, or worse, backed symbols
    or constants (e.g. if the original unbacked was replaced/specialized). Additionally this created problems with
    de/serialized programs, since we didn't comprehensively serialize ShapeEnv/unbacked renamings/node bindings.

    This pass attempts a simpler way of handling these for export, by throwing away the previously computed bindings, and re-running
    the pattern match used in compute_unbacked_bindings. This ensures we keep the original symbols contained in the example values,
    or delete bindings if they've been replaced/specialized.
    """
    from torch._export.utils import _get_shape_env_from_gm
    from torch.fx.experimental.symbolic_shapes import _free_unbacked_symbols_with_path
    from torch.utils._sympy.symbol import symbol_is_type, SymT

    if (shape_env := _get_shape_env_from_gm(gm)) is None:
        return

    base_unbacked_symbols = {
        symbol
        for symbol in shape_env.var_to_range
        if symbol_is_type(symbol, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT))
        and symbol not in shape_env.unbacked_renamings
    }
    for node in gm.graph.nodes:
        node.meta.pop("unbacked_bindings", None)
        if (val := node.meta.get("val")) is not None and (
            unbacked_bindings := _free_unbacked_symbols_with_path(
                val,
                (),
                shape_env=shape_env,
                pending=base_unbacked_symbols,
                simplify=True,
            )
        ):
            node.meta["unbacked_bindings"] = unbacked_bindings


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
    _prettify_placeholder_names=True,
) -> ATenExportArtifact:
    """
    This is a helper function that is shared between export_to_aten_ir and export_to_aten_ir_make_fx
    to produce the aten artifact. (export compatible graph module + signature)

    It does:
    1. Applies runtime assertion pass
    2. Recompute unbacked_bindings pass
    3. Populate meta val when missing
    4. Lift constants as placeholders
    5. Replace raw autograd and autocast ops with HOPs
    6. Prettify names for placeholders
    7. Preserve requires_grad value on node meta val
    """
    # Run runtime asserts pass before creating input/output specs, since size-related CSE/DCE might affect output signature.
    # Overwrite output specs afterwards.
    flat_fake_args = pytree.tree_leaves((fake_args, fake_kwargs))
    gm, graph_signature = apply_runtime_assertion_pass(gm, graph_signature)

    # Simplify unbacked_bindings by recomputing them.
    # Useful for any pass that's interpreter-based and might call rebind_unbacked(),
    # e.g. AOTAutograd in this case.
    _replace_unbacked_bindings(gm)

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
    constants = _materialize_and_lift_constants(
        gm, export_graph_signature, constant_attrs
    )

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
    if _prettify_placeholder_names:
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
    # Don't want to change the convention of previous call.
    param_buffer_table_reverse = {v: k for k, v in param_buffer_table.items()}

    # Replace state dict attr names with the fqn
    for name, _ in list(
        chain(
            original_module.named_parameters(remove_duplicate=False),
            # pyrefly: ignore [bad-argument-type]
            original_module.named_buffers(remove_duplicate=False),
        )
    ):
        if name in param_buffer_table_reverse:
            dynamo_name = param_buffer_table_reverse[name]
            param = torch.fx.graph_module._get_attr(traced_module, dynamo_name)
            torch.fx.graph_module._assign_attr(param, traced_module, name)
            torch.fx.graph_module._del_attr(traced_module, dynamo_name)

    # Replace graph getattr nodes with the correct name
    for node in traced_module.graph.nodes:
        if node.op == "get_attr":
            attr_name = node.target
            if attr_name in param_buffer_table:
                node.target = param_buffer_table[attr_name]

    traced_module.recompile()


def _get_module_hierarchy(mod: torch.nn.Module) -> dict[str, str]:
    return {
        name: type(m).__name__ for name, m in mod.named_modules(remove_duplicate=False)
    }


def _make_module_call_graph(
    in_spec: TreeSpec,
    out_spec: TreeSpec,
    module_call_signatures: dict[str, ModuleCallSignature],
    forward_arg_names: Optional[list[str]] = None,
) -> list[ModuleCallEntry]:
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


class _ExportModuleSpecTrackerDict(dict):
    pass


def _export_to_torch_ir(
    f: Callable,
    args: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]] = None,
    *,
    preserve_module_call_signature: tuple[str, ...] = (),
    disable_constraint_solver: bool = False,
    prefer_deferred_runtime_asserts_over_guards: bool = False,
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

    # Map ints to a wrapper structure to help us mark it as dynamic, if it is
    # dynamic. We will unwrap ints in fakify later.
    args, kwargs = pytree.tree_map_only(int, _IntWrapper, (args, kwargs))

    combined_args = _combine_args(f, args, kwargs)
    _check_dynamic_shapes(combined_args, dynamic_shapes)
    constraints = _process_dynamic_shapes(combined_args, dynamic_shapes)

    # Unwrap static ints -- in the case where we have an empty graph
    # containing just integer computation, dynamo will run its generated
    # bytecode with these args/kwargs, which will error because we cannot
    # directly apply int operations on IntWrapper. So we will just unwrap
    # them here.
    args, kwargs = pytree.tree_map_only(
        _IntWrapper,
        lambda a: a.val
        if a.dynamism is None or a.dynamism.type == _DimHintType.STATIC
        else a,
        (args, kwargs),
    )

    dynamo_cfg = dataclasses.replace(
        DEFAULT_EXPORT_DYNAMO_CONFIG,
        prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
    )

    def use_legacy_dynamo_graph_capture() -> bool:
        return bool(
            constraints  # dynamic shape
            or dynamic_shapes  # dynamic shape
            or isinstance(f, torch.fx.GraphModule)  # retracing
            or preserve_module_call_signature  # unflatten
            or torch._functorch.config.fake_tensor_propagate_real_tensors  # draft
            or torch._export.config.use_legacy_dynamo_graph_capture
        )

    with torch._dynamo.config.patch(dataclasses.asdict(dynamo_cfg)):
        try:
            module_call_specs: dict[str, dict[str, pytree.TreeSpec]] = (
                _ExportModuleSpecTrackerDict()
            )
            ctx = nullcontext()
            if not isinstance(f, torch.fx.GraphModule):
                ctx = _wrap_submodules(  # type: ignore[assignment]
                    f, preserve_module_call_signature, module_call_specs
                )
            with ctx, _ignore_backend_decomps():
                if torch._export.config.use_new_tracer_experimental:
                    from torch._dynamo.functional_export import (
                        _dynamo_graph_capture_for_export,
                        dynamo_graph_capture_for_export,
                    )

                    if use_legacy_dynamo_graph_capture():
                        dynamo_graph_capture = _dynamo_graph_capture_for_export(
                            f, constraints=constraints, dynamic_shapes=dynamic_shapes
                        )
                    else:
                        dynamo_graph_capture = dynamo_graph_capture_for_export(f)
                    # We can't serialize entire fake mode yet, so this is to make sure
                    # things like copy.deepcopy(ep.graph_module) not crash.
                    # see test_export.py::test_custom_tag_metadata_re_export
                    # Once we delete the old strict export, we can use
                    gm_torch_level = dynamo_graph_capture(*args, **kwargs)
                    # We can't serialize entire fake mode yet, so this is to make sure
                    # things like copy.deepcopy(ep.graph_module) not crash.
                    # see test_export.py::test_custom_tag_metadata_re_export
                    # Once we delete the old strict export, we can use this fake mode in the
                    # subsequent logic when lowering to aten IR.
                    del gm_torch_level.meta["fake_mode"]

                else:
                    gm_torch_level, _ = torch._dynamo.export(
                        f,
                        dynamic_shapes=dynamic_shapes,  # type: ignore[arg-type]
                        constraints=constraints,  # type: ignore[arg-type]
                        assume_static_by_default=True,
                        tracing_mode="symbolic",
                        disable_constraint_solver=disable_constraint_solver,
                        prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
                        _log_export_usage=_log_export_usage,
                        same_signature=same_signature,
                    )(
                        *args,
                        **kwargs,
                    )
                    gm_torch_level.meta["module_call_specs"] = module_call_specs
        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e))  # noqa: B904
        except GuardOnDataDependentSymNode as e:
            raise UserError(  # noqa: B904
                UserErrorType.ANTI_PATTERN,
                f"Consider annotating your code using torch._check*(). {str(e)}",
                case_name="constrain_as_size_example",
            )

    if isinstance(f, torch.nn.Module) and restore_fqn:
        _restore_state_dict(f, gm_torch_level)

    return gm_torch_level


def _aot_export_joint_with_descriptors(
    stack,
    mod,
    args,
    *,
    kwargs,
    decompositions,
    fake_params_buffers,
    _record_nn_module_stack=True,
):
    from torch._functorch._aot_autograd.graph_compile import aot_stage2_export
    from torch._functorch._aot_autograd.input_output_analysis import (
        create_graph_signature,
    )

    joint_with_descriptors = aot_export_joint_with_descriptors(
        stack,
        mod,
        args,
        kwargs=kwargs,
        decompositions=decompositions,
        _record_nn_module_stack=_record_nn_module_stack,
    )
    # Convert JointWithDescriptors to graph module and ViewAndMutationMeta
    gm, fw_metadata = aot_stage2_export(
        joint_with_descriptors._aot_state,
        joint_with_descriptors._aot_graph_capture,
    )

    assert isinstance(gm, torch.fx.GraphModule)

    # Create GraphSignature from the metadata
    graph_signature = create_graph_signature(
        gm,
        fw_metadata,
        joint_with_descriptors.in_spec,
        joint_with_descriptors.out_spec,
        user_args_flat=pytree.tree_leaves((args, kwargs)),
        params_and_buffers_flat=list(fake_params_buffers.values()),
        param_names=joint_with_descriptors.params_spec,
        buffer_names=joint_with_descriptors.buffers_spec,
        trace_joint=False,
        num_user_fw_outs=None,
        loss_index=None,
    )
    return gm, graph_signature


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
    _prettify_placeholder_names: bool = True,
    decompose_custom_triton_ops: bool = False,
) -> ATenExportArtifact:
    custom_triton_ops_decomposition_ctx = (
        nullcontext
        if decompose_custom_triton_ops
        else _disable_custom_triton_op_functional_decomposition
    )
    # This _reparameterize_module makes sure inputs and module.params/buffers have the same fake_mode,
    # otherwise aot_export_module will error out because it sees a mix of fake_modes.
    # And we want aot_export_module to use the fake_tensor mode in dynamo to keep the pipeline easy to reason about.
    with ExitStack() as stack:
        stack.enter_context(
            torch.nn.utils.stateless._reparametrize_module(
                mod,
                fake_params_buffers,
                tie_weights=True,
                strict=True,
                stack_weights=True,
            )
        )
        stack.enter_context(_ignore_backend_decomps())
        stack.enter_context(_compiling_state_context())
        stack.enter_context(custom_triton_ops_decomposition_ctx())
        stack.enter_context(torch.no_grad())

        gm, graph_signature = transform(_aot_export_joint_with_descriptors)(
            stack,
            mod,
            fake_args,
            kwargs=fake_kwargs,
            decompositions=decomp_table,
            fake_params_buffers=fake_params_buffers,
            _record_nn_module_stack=True,
        )

    def _maybe_fixup_gm_and_output_node_meta(old_gm, new_gm):
        if isinstance(old_gm, torch.fx.GraphModule):
            if hasattr(old_gm, "meta"):
                new_gm.meta.update(old_gm.meta)
            old_output_node = list(old_gm.graph.nodes)[-1]
            new_output_node = list(new_gm.graph.nodes)[-1]
            assert old_output_node.op == "output" and new_output_node.op == "output"
            # make sure we don't override any meta
            if "desc" in new_output_node.meta:
                del new_output_node.meta["desc"]
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
        _prettify_placeholder_names=_prettify_placeholder_names,
    )


def _get_forward_arg_names(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]] = None,
) -> list[str]:
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

    names: list[str] = []
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


def _get_non_persistent_buffers(mod: torch.nn.Module) -> set[str]:
    """
    Returns set of non-persistent buffers in a module and its submodules.
    """
    result: set[str] = set()
    for name, m in mod.named_modules(remove_duplicate=False):
        if name:
            result.update(f"{name}.{b}" for b in m._non_persistent_buffers_set)
        else:
            result.update(m._non_persistent_buffers_set)
    return result


def _rewrite_dynamo_tensor_constants(
    orig_mod_buffers: set[torch.Tensor],
    traced_mod_buffers: dict[str, torch.Tensor],
    graph_signature: ExportGraphSignature,
    constants: dict[str, _ConstantAttributeType],
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
    constants: dict[str, _ConstantAttributeType],
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


def get_ep_stats(ep: ExportedProgram) -> dict[str, Any]:
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


_EXPORT_FLAGS: Optional[set[str]] = None
_EXPORT_MODULE_HIERARCHY: Optional[dict[str, str]] = None


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

            if hasattr(e, "partial_fx_graph"):
                print(
                    e.partial_fx_graph,
                    file=sys.stderr,
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


def _get_original_state_dict(mod: torch.nn.Module) -> dict[str, Any]:
    # Explicitly not calling mode.state_dict() as we do not want the module state for serialization
    # but the running module state so we can always match by id() the entries here with the graph inputs
    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    original_state_dict = named_parameters | named_buffers

    non_persistent_buffers = _get_non_persistent_buffers(mod)
    for k in non_persistent_buffers:
        original_state_dict.pop(k, None)

    return original_state_dict


def _process_export_inputs(
    mod: torch.nn.Module,
    args: tuple[object, ...],
    kwargs: Optional[dict[str, object]],
    dynamic_shapes: Optional[
        Union[
            _DynamicShapesSpec,
            torch.export.AdditionalInputs,
            torch.export.ShapesCollection,
        ]
    ],
) -> tuple[
    tuple[object, ...],
    dict[str, object],
    TreeSpec,
    Optional[_DynamicShapesSpec],
    Callable[[ExportedProgram], None],
]:
    """
    Process and validate export inputs for the torch.export API.

    This function validates the input arguments, normalizes kwargs, computes input tree specs,
    and handles special dynamic shapes cases like AdditionalInputs and ShapesCollection.

    Args:
        mod: The PyTorch module to be exported.
        args: Tuple of example positional inputs for the module.
        kwargs: Optional dictionary of example keyword inputs.
        dynamic_shapes: Optional specification for dynamic shapes. Can be:
            - dict mapping argument names to dynamic shape specifications
            - tuple/list specifying dynamic shapes for each input in order
            - torch.export.AdditionalInputs object with verification callback
            - torch.export.ShapesCollection object

    Returns:
        A tuple containing:
        - args: Validated tuple of positional inputs
        - kwargs: Normalized dictionary of keyword inputs (empty dict if None was passed)
        - original_in_spec: TreeSpec representing the flattened input structure
        - dynamic_shapes: Processed dynamic shapes specification
        - verify_additional_inputs: Callback function for additional input verification

    Raises:
        UserError: If args is not a tuple.
    """
    if not isinstance(args, tuple):
        raise UserError(
            UserErrorType.INVALID_INPUT,
            f"Expecting `args` to be a tuple of example positional inputs, got {type(args)}",
        )
    kwargs = kwargs if kwargs is not None else {}
    if pytree.is_namedtuple_instance(args):
        args = tuple(args)

    _, original_in_spec = pytree.tree_flatten((args, kwargs))

    verify_additional_inputs: Callable[[ExportedProgram], None]
    out_dynamic_shapes: Optional[_DynamicShapesSpec]
    if isinstance(dynamic_shapes, torch.export.AdditionalInputs):
        verify_additional_inputs = dynamic_shapes.verify  # type: ignore[assignment]
        out_dynamic_shapes = dynamic_shapes.dynamic_shapes(mod, args, kwargs)  # type: ignore[assignment]
    else:
        verify_additional_inputs = lambda ep: None  # noqa: E731
        if isinstance(dynamic_shapes, torch.export.ShapesCollection):
            out_dynamic_shapes = dynamic_shapes.dynamic_shapes(mod, args, kwargs)  # type: ignore[assignment]
        else:
            out_dynamic_shapes = dynamic_shapes

    return args, kwargs, original_in_spec, out_dynamic_shapes, verify_additional_inputs


def _get_module_call_graph(
    export_artifact: ExportArtifact,
    preserve_module_call_signature: tuple[str, ...],
    strict_mode_export: bool,
    forward_arg_names: Optional[list[str]] = None,
) -> tuple[torch.fx.GraphModule, list[ModuleCallEntry]]:
    """
    In-place modify the graph module in export_artifact, remove _export_tracepoint nodes and
    return module_call_graph.
    """
    gm: torch.fx.GraphModule = export_artifact.aten.gm
    export_graph_signature: ExportGraphSignature = export_artifact.aten.sig
    module_call_specs: dict[str, dict[str, TreeSpec]] = (
        export_artifact.module_call_specs
    )
    in_spec: TreeSpec = export_artifact.in_spec
    out_spec: TreeSpec = export_artifact.out_spec

    # Make module signatures.
    module_call_signatures: dict[str, ModuleCallSignature] = {}
    for fqn, specs in module_call_specs.items():
        mod_fqn = _strip_root(fqn) if not strict_mode_export else fqn
        module_call_signatures[mod_fqn] = ModuleCallSignature(
            inputs=[],
            outputs=[],
            in_spec=specs["in_spec"],
            out_spec=specs["out_spec"],
            forward_arg_names=None,  # we only propagate forward_arg_names for the top level module
        )

    if len(preserve_module_call_signature) > 0:
        if not strict_mode_export:
            _rewrite_tracepoint_node(gm)
        res = CollectTracepointsPass(module_call_signatures, export_graph_signature)(gm)
        assert res is not None
        gm = res.graph_module

    assert _EXPORT_MODULE_HIERARCHY is not None
    module_call_graph = _make_module_call_graph(
        in_spec,
        out_spec,
        module_call_signatures,
        forward_arg_names,
    )
    return gm, module_call_graph


def _get_range_constraints(
    mod: torch.nn.Module,
    export_artifact: ExportArtifact,
    args,
    kwargs,
    dynamic_shapes,
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
    combined_args = _combine_args(mod, args, kwargs)

    # This is because we trace based on the kwargs passed in from user
    # not based on the signature. I feel it would be better to just enforce
    # one ordering at the start of tracing to avoid confusions, but that is
    # bigger refactor, so do this to unblock for now.
    combined_args_traced_order = {}
    for arg in combined_args:
        if arg not in kwargs:
            combined_args_traced_order[arg] = combined_args[arg]

    for key in kwargs:
        combined_args_traced_order[key] = kwargs[key]

    combined_args = combined_args_traced_order

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


def _strict_export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]],
    preserve_module_call_signature: tuple[str, ...],
    orig_in_spec: TreeSpec,
    prefer_deferred_runtime_asserts_over_guards: bool,
    _to_aten_func: Callable,
) -> ExportArtifact:
    """
    _to_aten_func can either be `_export_to_aten_ir_make_fx` or `_export_to_aten_ir`
    """

    gm_torch_level = _export_to_torch_ir(
        mod,
        args,
        kwargs,
        dynamic_shapes,
        preserve_module_call_signature=preserve_module_call_signature,
        restore_fqn=False,  # don't need to restore because we will do it later
        prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
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
                assert dynamo_fake_mode is not None, (
                    "Cannot find dynamo_fake_mode. This could be due to the exported graph module have no placeholders."
                )
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
        out_spec = pytree.treespec_tuple([out_spec])

    orig_arg_names = gm_torch_level.graph._codegen.pytree_info.orig_args  # type: ignore[attr-defined]

    gm_torch_level.graph._codegen.pytree_info = _PyTreeInfo(
        orig_arg_names,
        gm_torch_level._in_spec,
        out_spec,
    )
    gm_torch_level.recompile()

    _normalize_nn_module_stack(gm_torch_level, type(mod))

    params_buffers_to_node_meta = _collect_param_buffer_metadata(gm_torch_level)

    # When aot_export lifts the params, we lose metadata (e.g. source_fn_stack, stack_trace)
    # from the param nodes as they are treated as fresh inputs
    # Therefore, we manually extract them before calling into aot_export
    # params_buffers_to_node_meta = _collect_param_buffer_metadata(gm_torch_level)

    constant_attrs = _gather_constant_attrs(mod)
    param_buffer_table: dict[str, str] = _get_param_buffer_mapping(mod, gm_torch_level)

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

    tx = TracingContext(dynamo_fake_mode)
    with (
        dynamo_fake_mode,
        tracing(tx),
        mock.patch.object(dynamo_fake_mode, "allow_non_fake_inputs", True),
    ):
        aten_export_artifact = _to_aten_func(
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
        in_spec=orig_in_spec,
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
    def _make_fx_helper(stack, mod, args, kwargs, **flags):
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

        params_buffers_args: list[Any] = []
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
                assigned_buffers: dict[str, str] = {}
                hook = register_buffer_assignment_hook(
                    non_strict_root, assigned_buffers
                )

            def custom_getattribute(self, attr, *, original_getattr, attrs_to_proxy):
                """
                The idea here is that we override subclass getattr methods to proxy
                inner tensors and metadata. Because of infinite loop shenanigans, we have
                to manually construct the getattr proxy nodes without relying on torch function
                system.
                """
                out = original_getattr(self, attr)
                if attr in attrs_to_proxy:
                    if torch._C._is_torch_function_mode_enabled():
                        if isinstance(out, torch.Tensor):
                            # When we get here there is no guarantee that we will hit the
                            # PreDispatchTorchFunctionMode, so we manually peak into the torch
                            # function mode list and tweak the PreDispatchTorchFunctionMode.
                            # This has side effect of proxying stuff like
                            # proxy.node.meta["val"] = extract_val(val) because at that time, torch function
                            # mode is still active. It seems bad to turn it off inside proxy_tensor.py, so
                            # I guess we will just rely on DCE for now to remove extra stuff like detach
                            torch_function_mode_stack = (
                                torch.overrides._get_current_function_mode_stack()
                            )
                            for mode in torch_function_mode_stack:
                                if isinstance(mode, PreDispatchTorchFunctionMode):
                                    tracer = mode.tracer
                                    proxy = get_proxy_slot(self, tracer).proxy
                                    inner_proxy = tracer.create_proxy(
                                        "call_function",
                                        torch.ops.export.access_subclass_inner_tensor.default,
                                        (proxy, attr),
                                        {},
                                    )
                                    track_tensor_tree(
                                        out, inner_proxy, constant=None, tracer=tracer
                                    )
                return out

            @contextmanager
            def override_getattribute_for_subclasses(args):
                """
                Context manager that temporarily monkey patches
                tensor.__getattribute__ so that we can intercept it at
                torch_function layer.
                """

                # Dictionary that tracks subclass type to original getattr function
                # and the attributes we can proxy.
                tensor_type_to_old_getattribute: dict[
                    type[torch.Tensor], tuple[Callable, set[str]]
                ] = {}
                for arg in args:
                    subclass_types_to_instances: dict[
                        type[torch.Tensor], list[type[torch.Tensor]]
                    ] = get_subclass_typing_container(arg)
                    for subclass_type in subclass_types_to_instances:
                        if subclass_type not in tensor_type_to_old_getattribute:
                            assert len(subclass_types_to_instances[subclass_type]) > 0
                            instance = subclass_types_to_instances[subclass_type][0]
                            # Query subclass specific attrs
                            attrs_to_proxy = set(dir(instance)) - set(dir(torch.Tensor))
                            tensor_type_to_old_getattribute[subclass_type] = (
                                subclass_type.__getattribute__,  # type: ignore[attr-defined]
                                attrs_to_proxy,
                            )

                try:
                    for k, (
                        old_getattr,
                        attrs_to_proxy,
                    ) in tensor_type_to_old_getattribute.items():
                        custom = functools.partialmethod(
                            custom_getattribute,
                            original_getattr=old_getattr,
                            attrs_to_proxy=attrs_to_proxy,
                        )
                        k.__getattribute__ = custom  # type: ignore[assignment, attr-defined]
                    yield
                finally:
                    for k, (old_getattr, _) in tensor_type_to_old_getattribute.items():
                        k.__getattribute__ = old_getattr  # type: ignore[method-assign, attr-defined]

            @contextmanager
            def _maybe_restore_grad_state():
                """
                When pre-dispatch export accidentally change grad state, we restore it back.
                This can happen when we are calling torch._C._set_grad_enabled directly in the
                forward.
                """
                old_state = torch.is_grad_enabled()
                try:
                    yield
                finally:
                    torch._C._set_grad_enabled(old_state)

            with (
                ctx,
                override_getattribute_for_subclasses(flat_args),
                _maybe_restore_grad_state(),
            ):
                gm = make_fx(
                    wrapped_fn,
                    record_module_stack=True,
                    pre_dispatch=True,
                )(*flat_args)

            if non_strict_root is not None:
                input_names = _graph_input_names(gm)
                buffer_input_names = {
                    name: input_names[param_len + i]
                    for i, (name, buf) in enumerate(non_strict_root._buffers.items())
                    if buf is not None
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

            def _is_impure(node):
                if node.op == "call_function" and node.target in (
                    # In export, we ignore any op that is related to
                    # eager mode profiling call. The expectation is
                    # that either runtimes provide their own profiling
                    # OR user wrap the compiled region on a profiling in
                    # later stage.
                    torch.ops.profiler._record_function_enter.default,
                    torch.ops.profiler._record_function_enter_new.default,
                    torch.ops.profiler._record_function_exit._RecordFunction,
                    # In theory, we could fix this dead detach and getattr nodes
                    # from subclass tensors if we carefully rewrite track_tensor_tree
                    # in a way that it doesn't do any tensor methods.
                    torch.ops.aten.detach.default,
                    torch.ops.export.access_subclass_inner_tensor.default,
                ):
                    return False
                return True

            gm.graph.eliminate_dead_code(_is_impure)

        # create graph signature
        assert out_spec.spec is not None, "out_spec.spec is None!"
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
            parameters_to_mutate={},
            user_inputs_to_mutate={},
            in_spec=in_spec,
            out_spec=out_spec.spec,
            backward_signature=None,
            input_tokens=[],
            output_tokens=[],
        )
        return gm, sig

    # This _reparameterize_module makes sure inputs and module.params/buffers have the same fake_mode,
    # otherwise aot_export_module will error out because it sees a mix of fake_modes.
    # And we want aot_export_module to use the fake_tensor mode in dynamo to keep the pipeline easy to reason about.
    with ExitStack() as stack:
        stack.enter_context(
            torch.nn.utils.stateless._reparametrize_module(
                mod,
                fake_params_buffers,
                tie_weights=True,
                strict=True,
                stack_weights=True,
            )
        )
        stack.enter_context(_ignore_backend_decomps())
        stack.enter_context(_compiling_state_context())
        gm, graph_signature = transform(_make_fx_helper)(
            stack,
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
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if index >= num_params_buffers:
                user_arg = flat_args[index - num_params_buffers]
                if not isinstance(user_arg, torch.Tensor):
                    node.meta["val"] = user_arg
            index += 1


def _find_node(gm: torch.fx.GraphModule, name: str) -> torch.fx.Node:
    return next(iter(node for node in gm.graph.nodes if node.name == name))


def _non_strict_export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]],
    preserve_module_call_signature: tuple[str, ...],
    orig_in_spec: TreeSpec,
    prefer_deferred_runtime_asserts_over_guards: bool,
    _to_aten_func: Callable,
) -> ExportArtifact:
    """
    _to_aten_func can either be `_export_to_aten_ir_make_fx` or `_export_to_aten_ir`
    """

    out_spec: Optional[TreeSpec] = None
    in_spec: Optional[TreeSpec] = None

    module_call_specs: dict[str, dict[str, pytree.TreeSpec]] = {}

    def _tuplify_outputs(aot_export):
        def _aot_export_non_strict(stack, mod, args, *, kwargs=None, **flags):
            kwargs = kwargs or {}

            class Wrapper(torch.nn.Module):
                def __init__(self, mod):
                    super().__init__()
                    self._export_root = mod

                def forward(self, *args, **kwargs):
                    nonlocal out_spec
                    nonlocal in_spec
                    mod = self._export_root
                    _, in_spec = pytree.tree_flatten((args, kwargs))
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
            ctx = nullcontext()
            if not isinstance(mod, torch.fx.GraphModule):
                ctx = _wrap_submodules(  # type: ignore[assignment]
                    wrapped_mod, new_preserved_call_signatures, module_call_specs
                )
            with ctx:
                gm, sig = aot_export(stack, wrapped_mod, args, kwargs=kwargs, **flags)
            log.debug("Exported program from AOTAutograd:\n%s", gm)

            sig.parameters = pytree.tree_map(_strip_root, sig.parameters)
            sig.buffers = pytree.tree_map(_strip_root, sig.buffers)
            sig.inputs_to_buffers = pytree.tree_map(_strip_root, sig.inputs_to_buffers)
            sig.inputs_to_parameters = pytree.tree_map(
                _strip_root, sig.inputs_to_parameters
            )
            sig.buffers_to_mutate = pytree.tree_map(_strip_root, sig.buffers_to_mutate)
            sig.parameters_to_mutate = pytree.tree_map(
                _strip_root, sig.parameters_to_mutate
            )

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
        prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,  # for shape env initialization
    )

    fake_params_buffers = _fakify_params_buffers(fake_mode, mod)

    def _produce_guards_callback(gm):
        return produce_guards_and_solve_constraints(
            fake_mode=fake_mode,
            gm=gm,
            dynamic_shapes=dynamic_shapes,
            equalities_inputs=equalities_inputs,
            original_signature=original_signature,
        )

    tx = TracingContext(fake_mode)

    # We also need to attach dynamo configs as these will be used in HOOs that
    # use torch.compile, like cond
    dynamo_config = dataclasses.asdict(DEFAULT_EXPORT_DYNAMO_CONFIG)
    dynamo_config["do_not_emit_runtime_asserts"] = (
        False  # We want to emit runtime asserts
    )

    with (
        fake_mode,
        _NonStrictTorchFunctionHandler(),
        tracing(tx),
        torch._dynamo.config.patch(dynamo_config),
    ):
        with (
            _fakify_script_objects(mod, fake_args, fake_kwargs, fake_mode) as (
                patched_mod,
                new_fake_args,
                new_fake_kwargs,
                new_fake_constant_attrs,
                map_fake_to_real,
            ),
            _fakify_module_inputs(fake_args, fake_kwargs, fake_mode),
            _override_builtin_ops(),
        ):
            # _to_aten_func is _export_to_aten_ir when using the default non-strict export
            # We need to pass positional args correctly
            aten_export_artifact = _to_aten_func(
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
    assert in_spec is not None

    return ExportArtifact(
        aten=aten_export_artifact,
        in_spec=in_spec,
        out_spec=out_spec,
        fake_mode=fake_mode,
        module_call_specs=module_call_specs,
    )


@_log_export_wrapper
@_disable_prexisiting_fake_mode
def _export_for_training(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]] = None,
    *,
    strict: bool = True,
    preserve_module_call_signature: tuple[str, ...] = (),
    prefer_deferred_runtime_asserts_over_guards: bool = False,
) -> ExportedProgram:
    global _EXPORT_MODULE_HIERARCHY
    _EXPORT_MODULE_HIERARCHY = _get_module_hierarchy(mod)

    (
        args,
        kwargs,
        orig_in_spec,
        dynamic_shapes,
        verify_additional_inputs,
    ) = _process_export_inputs(mod, args, kwargs, dynamic_shapes)

    original_state_dict = _get_original_state_dict(mod)

    has_ambient_mode = False
    if not strict:
        flat_args, _ = pytree.tree_flatten((args, kwargs))
        has_ambient_mode = torch._guards.detect_fake_mode(flat_args) is not None

    # Call the appropriate export function based on the strictness of tracing.
    export_func = _strict_export if strict else _non_strict_export

    if not strict and torch._export.config.detect_non_strict_fake_tensor_leaks:
        from torch._subclasses.fake_tensor import fake_tensor_tls

        fake_tensor_tls.non_strict_export_fake_tensor_tracker.clear()

    export_artifact = export_func(
        mod=mod,
        args=args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        preserve_module_call_signature=preserve_module_call_signature,
        orig_in_spec=orig_in_spec,
        prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
        _to_aten_func=_export_to_aten_ir_make_fx,
    )

    # If we are tracing with fake inputs, it is expected to
    # see fake tensor constants.
    if not strict and not has_ambient_mode:
        for const, val in export_artifact.aten.constants.items():
            if isinstance(
                val, torch._subclasses.fake_tensor.FakeTensor
            ) and _is_bogus_const_name(const):
                error_msg = (
                    f"We found a fake tensor in the exported program constant's list. "
                    f"This typically means our tracing system encountered an op that "
                    f"we can't trace through. For the potential source, you can refer to "
                    f"following model attribute: {const}. "
                    f"Please file an issue on github. "
                )
                if torch._export.config.error_on_lifted_constant_tensors:
                    raise RuntimeError(error_msg)
                else:
                    warnings.warn(error_msg, stacklevel=2)

    export_graph_signature = export_artifact.aten.sig

    forward_arg_names = _get_forward_arg_names(mod, args, kwargs)
    inline_constraints = _get_inline_constraints(export_artifact.fake_mode)
    # The unbacked symint symbols are updated in aot_export
    # so we serialize them here instead of inside dynamo.
    # Note: _get_range_constraints depends on "inline_constraints" to be set.
    export_artifact.aten.gm.meta["inline_constraints"] = inline_constraints
    range_constraints = _get_range_constraints(
        mod,
        export_artifact,
        args,
        kwargs,
        dynamic_shapes,
    )
    # The returned the gm is in-place modified
    gm, module_call_graph = _get_module_call_graph(
        export_artifact,
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

    verify_additional_inputs(exported_program)

    if not strict and torch._export.config.detect_non_strict_fake_tensor_leaks:
        # See NOTE [export non-strict fake tensor leak detection]
        from torch._subclasses.fake_tensor import fake_tensor_tls
        from torch.fx.experimental.proxy_tensor import (
            _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT,
        )

        active_fakes = fake_tensor_tls.non_strict_export_fake_tensor_tracker
        legit_leak: weakref.WeakSet = find_legit_leaks_from_referrers(active_fakes)
        leak_sources: list[str] = []
        if len(legit_leak) > 0:
            for fake_val in legit_leak:
                if id(fake_val) in _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT:
                    stack_trace = _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT[
                        id(fake_val)
                    ].meta.get("stack_trace", "<unknown stack trace>")

                    # Get shape and dtype info
                    shape_info = f"shape={fake_val.shape}, dtype={fake_val.dtype}"
                    leak_info = f"FakeTensor({shape_info}): {stack_trace}"
                    leak_sources.append(leak_info)

            # Format the warning message more nicely
            leak_details = "\n  ".join(leak_sources)
            warnings.warn(
                f"Detected {len(legit_leak)} fake tensors that are still alive after export.\n"
                f"This is likely result of torch.export.export not being able to track side effects "
                f"that is happening outside of model scope.\n\n"
                f"Leaked tensors:\n  {leak_details}\n\n"
                f"Alternatively, please file a bug report to PyTorch team for further debugging help.",
                stacklevel=2,
            )

            del legit_leak

    return exported_program


@_log_export_wrapper
@_disable_prexisiting_fake_mode
def _export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]] = None,
    *,
    strict: bool = True,
    preserve_module_call_signature: tuple[str, ...] = (),
    pre_dispatch: bool = False,
    prefer_deferred_runtime_asserts_over_guards: bool = False,
) -> ExportedProgram:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a ExportedProgram.

    Args:
        mod: the `nn.Module` to trace.

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

        prefer_deferred_runtime_asserts_over_guards:
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
        An ExportedProgram containing the traced module.
    """

    from torch._utils_internal import export_training_ir_rollout_check

    global _EXPORT_FLAGS, _EXPORT_MODULE_HIERARCHY
    _EXPORT_MODULE_HIERARCHY = _get_module_hierarchy(mod)

    flags = set()
    flags.add("strict" if strict else "non_strict")
    flags.add("pre_dispatch" if pre_dispatch else "aot_dispatch")
    _EXPORT_FLAGS = flags

    log_export_usage(event="export.enter", flags=_EXPORT_FLAGS)

    dtrace_structured("export", payload_fn=lambda: "start!")

    # NOTE Export training IR rollout
    # Old export calls export._trace(pre_dispatch=True)
    # and there are still lot of internal/OSS callsites that
    # use export._trace(pre_dispatch=True) directly. Therefore,
    # it makes more sense to do the switch here.
    # export_training_ir_rollout_check returns True in OSS
    # while internally it returns False UNLESS otherwise specified.
    if pre_dispatch and export_training_ir_rollout_check():
        ep = _export_for_training(
            mod,
            args,
            kwargs,
            dynamic_shapes,
            strict=strict,
            preserve_module_call_signature=preserve_module_call_signature,
            prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
        )
        dtrace_structured("exported_program", payload_fn=lambda: str(ep))
        return ep

    (
        args,
        kwargs,
        original_in_spec,
        dynamic_shapes,
        verify_additional_inputs,
    ) = _process_export_inputs(mod, args, kwargs, dynamic_shapes)

    original_state_dict = _get_original_state_dict(mod)

    # Call the appropriate export function based on the strictness of tracing.
    export_func = _strict_export if strict else _non_strict_export

    export_artifact = export_func(  # type: ignore[operator]
        mod=mod,
        args=args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        preserve_module_call_signature=preserve_module_call_signature,
        orig_in_spec=original_in_spec,
        prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
        _to_aten_func=functools.partial(
            _export_to_aten_ir,
            pre_dispatch=pre_dispatch,
        ),
    )
    export_graph_signature: ExportGraphSignature = export_artifact.aten.sig

    forward_arg_names = _get_forward_arg_names(mod, args, kwargs)
    inline_constraints = _get_inline_constraints(export_artifact.fake_mode)
    # The unbacked symint symbols are updated in aot_export
    # so we serialize them here instead of inside dynamo.
    # Note: this step must be before _get_range_constraints.
    export_artifact.aten.gm.meta["inline_constraints"] = inline_constraints
    range_constraints = _get_range_constraints(
        mod,
        export_artifact,
        args,
        kwargs,
        dynamic_shapes,
    )
    gm, module_call_graph = _get_module_call_graph(
        export_artifact,
        preserve_module_call_signature,
        strict,
        forward_arg_names,
    )

    _verify_nn_module_stack(gm)
    _verify_stack_trace(gm)
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

    dtrace_structured("exported_program", payload_fn=lambda: str(exported_program))

    verify_additional_inputs(exported_program)
    return exported_program
