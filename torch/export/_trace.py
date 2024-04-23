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
from torch._dynamo.exc import UserError, UserErrorType
from torch._export.non_strict_utils import (
    make_constraints,
    make_fake_inputs,
    make_fake_params_buffers,
)
from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
    _AddRuntimeAssertionsForInlineConstraintsPass,
)
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass
from torch._export.passes.lift_constants_pass import (
    ConstantAttrMap,
    lift_constants_pass,
    rewrite_script_object_meta,
)
from torch._export.utils import placeholder_naming_pass, placeholder_prefixes
from torch._export.verifier import SpecViolationError
from torch._export.wrappers import _wrap_submodules
from torch._functorch.aot_autograd import aot_export_module
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch._utils_internal import log_export_usage
from torch.export.exported_program import OutputKind
from torch.fx._utils import first_call_function_nn_module_stack
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
    ShapeEnv,
)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts
from torch.utils._pytree import TreeSpec
from torch.utils._sympy.value_ranges import ValueRangeError

from ._safeguard import AutogradStateOpsFailSafeguard

from .dynamic_shapes import _process_constraints
from .exported_program import (
    _disable_prexisiting_fake_mode,
    ExportedProgram,
    InputKind,
    ModuleCallEntry,
    ModuleCallSignature,
)
from .graph_signature import (
    _sig_to_specs,
    ArgumentSpec,
    ConstantArgument,
    CustomObjArgument,
    ExportGraphSignature,
    SymIntArgument,
    TensorArgument,
    TokenArgument,
)


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


def _convert_input_to_fake(gm, args, kwargs):
    params_buffers = _get_params_buffers(gm)
    fake_inps: List[torch.Tensor] = []
    for node in gm.graph.nodes:
        if node.op == "placeholder" and "val" in node.meta:
            fake_val = node.meta["val"]
            if fake_val is not None and isinstance(fake_val, torch.Tensor):
                fake_inps.append(fake_val)

    if detected_fake_mode := detect_fake_mode(fake_inps):
        fake_mode = detected_fake_mode
    else:
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())

    if len(args) == 0 and len(kwargs) == 0:
        return (), {}, params_buffers, fake_mode

    count = 0

    def convert_to_fake(x):
        nonlocal count
        val = fake_inps[count]
        count += 1
        return val

    fake_args = pytree.tree_map_only(torch.Tensor, convert_to_fake, args)
    # TODO properly use the cached fake tensor
    fake_kwargs = pytree.tree_map_only(torch.Tensor, fake_mode.from_tensor, kwargs)
    fake_params_buffers = pytree.tree_map_only(
        torch.Tensor,
        functools.partial(fake_mode.from_tensor, static_shapes=True),
        params_buffers,
    )
    return fake_args, fake_kwargs, fake_params_buffers, fake_mode


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

    param_lookup: Dict[int, List[str]] = {}
    buffer_lookup: Dict[int, List[str]] = {}
    for name, param in original_module.named_parameters(remove_duplicate=False):
        param_lookup.setdefault(id(param), []).append(name)
    for name, buffer in original_module.named_buffers(remove_duplicate=False):
        buffer_lookup.setdefault(id(buffer), []).append(name)

    param_buffer_table: Dict[str, str] = {}
    for dynamo_name, dynamo_param in traced_module.named_parameters(
        remove_duplicate=False
    ):
        assert dynamo_name not in param_buffer_table
        if id(dynamo_param) in param_lookup:
            param_buffer_table[dynamo_name] = param_lookup[id(dynamo_param)].pop()

    for dynamo_name, dynamo_buffer in traced_module.named_buffers(
        remove_duplicate=False
    ):
        assert dynamo_name not in param_buffer_table
        if id(dynamo_buffer) in buffer_lookup:
            param_buffer_table[dynamo_name] = buffer_lookup[id(dynamo_buffer)].pop()

    return param_buffer_table


def _remap_constants(
    orig_constant_attrs: ConstantAttrMap,
    graph_signature: ExportGraphSignature,
    constants: Dict[str, Union[torch.Tensor, torch.ScriptObject]],
) -> None:
    """Rewrite the graph signature and constants table to use the FQN from the original module."""
    remap_table: Dict[str, str] = {}
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
            spec.target = remap_table.get(orig_target, orig_target)

            constant = constants[orig_target]
            del constants[orig_target]
            constants[spec.target] = constant


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
    module_hierarchy: Dict[str, str],
    in_spec: TreeSpec,
    out_spec: TreeSpec,
    module_call_signatures: Dict[str, ModuleCallSignature],
) -> List[ModuleCallEntry]:
    ret = [
        ModuleCallEntry(fqn=fqn, signature=module_call_signatures.get(fqn))
        for fqn in module_hierarchy
    ]
    assert ret[0].fqn == ""
    ret[0].signature = ModuleCallSignature(
        inputs=[], outputs=[], in_spec=in_spec, out_spec=out_spec
    )
    return ret


def _export_to_torch_ir(
    f: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]] = None,
    *,
    preserve_module_call_signature: Tuple[str, ...] = (),
    disable_constraint_solver: bool = False,
    restore_fqn: bool = True,
    _log_export_usage: bool = True,
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
                    _log_export_usage=_log_export_usage,
                )(
                    *args,
                    **kwargs,
                )
        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e))  # noqa: TRY200
        except GuardOnDataDependentSymNode as e:
            raise UserError(  # noqa: TRY200
                UserErrorType.ANTI_PATTERN,
                f"Consider annotating your code using torch._constrain_as_*(). {str(e)}",
                case_name="constrain_as_size_example",
            )

    gm_torch_level.meta["module_call_specs"] = module_call_specs

    if isinstance(f, torch.nn.Module) and restore_fqn:
        _restore_state_dict(f, gm_torch_level)

    return gm_torch_level


def _gather_constant_attrs(m: torch.nn.Module) -> ConstantAttrMap:
    """Search the module hierarchy, gathering up all tensor and ScriptObject constants.

    Returns a dictionary mapping hash(value) to the name of the constant. We
    have to abuse `hash` here unfortunately, see: [ScriptObject hash].
    """
    constants = ConstantAttrMap()
    buffers_parameters = set(m.buffers())
    buffers_parameters.update(m.parameters())

    def inner(m: torch.nn.Module, prefix_atoms: List[str], constants):
        for k, v in m.__dict__.items():
            if isinstance(v, (torch.Tensor, torch.ScriptObject)):
                if v in buffers_parameters:
                    # filter out buffers and parameters, leaving only constants
                    continue

                fqn = ".".join(prefix_atoms + [k])
                if v in constants:
                    raise ValueError(
                        f"Duplicate reference to constant attribute found: '{constants[v]}' and '{fqn}'."
                    )

                constants[v] = fqn
        for k, v in m.named_children():
            inner(v, prefix_atoms + [k], constants)

    inner(m, [], constants)
    return constants


def _export_non_strict(
    mod: torch.nn.Module,
    fake_args,
    fake_kwargs,
    fake_params_buffers,
    constant_attrs: ConstantAttrMap,
    *,
    transform=lambda x: x,  # TODO(zhxchen17) Revisit if this is needed later.
    pre_dispatch=False,
):
    # [NOTE] If the user is exporting under training mode, we want to detect if there is any
    # state change in the autograd global state and error. If the user is exporting under inference
    # mode, we don't care.
    is_grad_enabled = torch._C.is_grad_enabled()
    grad_safe_guard = (
        AutogradStateOpsFailSafeguard() if is_grad_enabled else nullcontext()
    )

    @contextmanager
    def _compiling_state_context():
        old_value = torch.compiler._is_compiling_flag
        try:
            torch.compiler._is_compiling_flag = True
            yield
        finally:
            torch.compiler._is_compiling_flag = old_value

    # This _reparametrize_module makes sure inputs and module.params/buffers have the same fake_mode,
    # otherwise aot_export_module will error out because it sees a mix of fake_modes.
    # And we want aot_export_module to use the fake_tensor mode in dynamo to keep the pipeline easy to reason about.
    with torch.nn.utils.stateless._reparametrize_module(
        mod, fake_params_buffers
    ), grad_safe_guard, _ignore_backend_decomps(), _compiling_state_context():  # type: ignore[attr-defined]
        gm, graph_signature = transform(aot_export_module)(
            mod,
            fake_args,
            trace_joint=False,
            pre_dispatch=pre_dispatch,
            kwargs=fake_kwargs,
        )
    # TODO unfortunately preserving graph-level metadata is not
    # working well with aot_export. So we manually copy it.
    # (The node-level meta is addressed above.)
    if isinstance(mod, torch.fx.GraphModule) and hasattr(mod, "meta"):
        gm.meta.update(mod.meta)

    if pre_dispatch:
        from torch._export.passes.replace_set_grad_with_hop_pass import (
            replace_set_grad_with_hop_pass,
        )

        gm = replace_set_grad_with_hop_pass(gm)

    # Remove nn_module_stack, stack_trace metadata from all placeholders/inputs nodes.
    for _mod in gm.modules():
        if not isinstance(_mod, torch.fx.GraphModule):
            continue
        for node in _mod.graph.nodes:
            if node.op in ["placeholder", "output"]:
                node.meta.pop("nn_module_stack", None)
                node.meta.pop("stack_trace", None)

    # NOTE: aot_export adds symint metadata for placeholders with int values;
    # since these become specialized, we replace such metadata with the original values
    flat_args = pytree.tree_leaves((fake_args, fake_kwargs))
    index = 0
    total_non_user_inputs = (
        len(graph_signature.parameters)
        + len(graph_signature.buffers)
        + len(graph_signature.input_tokens)
    )
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if index >= total_non_user_inputs:
                user_arg = flat_args[index - total_non_user_inputs]
                if not isinstance(user_arg, torch.Tensor):
                    node.meta["val"] = user_arg
            index += 1

    is_joint = graph_signature.backward_signature is not None

    def make_argument_spec(i, node) -> ArgumentSpec:
        if isinstance(node, (int, bool, float, type(None))):
            # For const outputs we just directly return this
            return ConstantArgument(name="", value=node)

        assert (
            "val" in node.meta
        ), f"{node} is not a constant or a node with a 'val' metadata field"
        val = node.meta["val"]
        if i < len(graph_signature.input_tokens):
            # TODO: We should be checking for a different type, once we add a new type
            return TokenArgument(name=node.name)
        elif isinstance(val, FakeTensor):
            return TensorArgument(name=node.name)
        elif isinstance(val, torch.SymInt):
            return SymIntArgument(name=node.name)
        elif isinstance(val, torch.ScriptObject):
            return CustomObjArgument(
                name=node.name, class_fqn=val._type().qualified_name()  # type: ignore[attr-defined]
            )
        else:
            # TODO: this branch is likely wrong, all permissible ConstantArgument type
            # should have been handled already
            return ConstantArgument(name=node.name, value=val)

    input_specs, output_specs = _sig_to_specs(
        user_inputs=set(graph_signature.user_inputs),
        inputs_to_parameters=graph_signature.inputs_to_parameters,  # type: ignore[arg-type]
        inputs_to_buffers=graph_signature.inputs_to_buffers,  # type: ignore[arg-type]
        user_outputs=set(graph_signature.user_outputs),  # type: ignore[arg-type]
        buffer_mutations=graph_signature.buffers_to_mutate,  # type: ignore[arg-type]
        user_input_mutations=graph_signature.user_inputs_to_mutate,  # type: ignore[arg-type]
        grad_params=graph_signature.backward_signature.gradients_to_parameters if is_joint else {},  # type: ignore[arg-type, union-attr]
        grad_user_inputs=graph_signature.backward_signature.gradients_to_user_inputs if is_joint else {},  # type: ignore[arg-type, union-attr]
        loss_output=graph_signature.backward_signature.loss_output if is_joint else None,  # type: ignore[arg-type, union-attr]
        inputs=[
            make_argument_spec(i, node)
            for i, node in enumerate(gm.graph.nodes)
            if node.op == "placeholder"
        ],
        outputs=[
            make_argument_spec(i, node)
            for i, node in enumerate(
                pytree.tree_leaves(next(iter(reversed(gm.graph.nodes))).args)
            )
        ],
        input_tokens=graph_signature.input_tokens,
        output_tokens=graph_signature.output_tokens,
    )
    export_graph_signature = ExportGraphSignature(
        input_specs=input_specs, output_specs=output_specs
    )

    constants = rewrite_script_object_meta(gm)
    constants.update(lift_constants_pass(gm, export_graph_signature, constant_attrs))

    # prettify names for placeholder nodes
    placeholder_naming_pass(
        gm,
        export_graph_signature,
        mod,
        fake_args,
        fake_kwargs,
        fake_params_buffers,
        constants,
    )

    @dataclasses.dataclass
    class _ExportedProgramNonStrict:
        gm: torch.fx.GraphModule
        sig: ExportGraphSignature
        constants: Dict[str, Union[torch.Tensor, torch._C.ScriptObject]]

    return _ExportedProgramNonStrict(
        gm,
        export_graph_signature,
        constants,
    )


def _get_params_buffers(mod: torch.nn.Module) -> Dict[str, torch.Tensor]:
    params_buffers: Dict[str, torch.Tensor] = {}
    for name, param in mod.named_parameters(remove_duplicate=False):
        params_buffers[name] = param

    for name, buffer in mod.named_buffers(remove_duplicate=False):
        params_buffers[name] = buffer
    return params_buffers


def _rewrite_dynamo_tensor_constants(
    orig_mod_buffers: Set[torch.Tensor],
    traced_mod_buffers: Dict[str, torch.Tensor],
    graph_signature: ExportGraphSignature,
    constants: Dict[str, Union[torch.Tensor, torch.ScriptObject]],
):
    """Dynamo erroneously marks tensor attributes on modules as a buffers.

    Rewrite them to be tensor constants.
    """
    for spec in graph_signature.input_specs:
        if spec.kind == InputKind.BUFFER:
            assert spec.target is not None
            value = traced_mod_buffers[spec.target]
            if value not in orig_mod_buffers:
                # This was a tensor constant erroneously marked as a buffer.
                # Convert it int oa constant in the graph signature, and add its
                # value to the constants table.
                spec.kind = InputKind.CONSTANT_TENSOR
                constants[spec.target] = value


def _rewrite_non_persistent_buffers(
    orig_mod: torch.nn.Module,
    graph_signature: ExportGraphSignature,
    constants: Dict[str, Union[torch.Tensor, torch.ScriptObject]],
):
    """Dynamo erroneously drops the persistent flag on buffers.

    Rewrite non-persistent buffers to reflect the original module.
    """
    state_dict = orig_mod.state_dict()
    for spec in graph_signature.input_specs:
        if spec.kind == InputKind.BUFFER:
            assert spec.target is not None
            if spec.target not in state_dict:
                assert spec.target not in constants
                spec.persistent = False
                constants[spec.target] = orig_mod.get_buffer(spec.target)


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
    for i, mod in enumerate([graph_module] + list(graph_module.modules())):
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


def _verify_placeholder_names(gm: torch.fx.GraphModule, sig: ExportGraphSignature):
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
            log_export_usage(
                event="export.error",
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

    Returns:
        An ExportedProgram containing the traced method.
    """
    from .dynamic_shapes import _process_dynamic_shapes

    if not isinstance(args, tuple):
        raise UserError(
            UserErrorType.INVALID_INPUT,
            f"Expecting `args` to be a tuple of example positional inputs, got {type(args)}",
        )

    global _EXPORT_FLAGS, _EXPORT_MODULE_HIERARCHY
    _EXPORT_MODULE_HIERARCHY = _get_module_hierarchy(mod)

    flags = set()
    flags.add("strict" if strict else "non_strict")
    flags.add("pre_dispatch" if pre_dispatch else "aot_dispatch")
    log_export_usage(event="export.enter", flags=flags)
    _EXPORT_FLAGS = flags

    kwargs = kwargs or {}
    _process_dynamic_shapes(mod, args, kwargs, dynamic_shapes)  # TODO(avik): remove

    constant_attrs = _gather_constant_attrs(mod)

    flat_args, orig_in_spec = pytree.tree_flatten((args, kwargs))

    if not strict:
        out_spec = None

        module_call_specs: Dict[str, Dict[str, pytree.TreeSpec]] = {}

        def strip_root(x):
            if isinstance(x, str) and x.startswith("_export_root"):
                stripped = x[len("_export_root") :]
                return stripped[1:] if stripped.startswith(".") else stripped
            return x

        def fixup_key(x):
            return "L__self__" + strip_root(x)

        def _tuplify_outputs(aot_export):
            def _aot_export_non_strict(mod, args, kwargs=None, **flags):
                kwargs = kwargs or {}

                class Wrapper(torch.nn.Module):
                    def __init__(self, mod):
                        super().__init__()
                        self._export_root = mod

                    def forward(self, *args, **kwargs):
                        nonlocal out_spec
                        if isinstance(self._export_root, torch.fx.GraphModule):
                            with torch.fx.traceback.preserve_node_meta():
                                tree_out = torch.fx.Interpreter(self._export_root).run(
                                    *args, **kwargs
                                )
                        else:
                            tree_out = self._export_root(*args, **kwargs)
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

                sig.parameters = pytree.tree_map(strip_root, sig.parameters)
                sig.buffers = pytree.tree_map(strip_root, sig.buffers)
                sig.inputs_to_buffers = pytree.tree_map(
                    strip_root, sig.inputs_to_buffers
                )
                sig.inputs_to_parameters = pytree.tree_map(
                    strip_root, sig.inputs_to_parameters
                )
                sig.buffers_to_mutate = pytree.tree_map(
                    strip_root, sig.buffers_to_mutate
                )
                for node in gm.graph.nodes:
                    if "nn_module_stack" in node.meta:
                        nn_module_stack = node.meta["nn_module_stack"]
                        node.meta["nn_module_stack"] = {
                            fixup_key(key): val
                            for key, val in pytree.tree_map(
                                strip_root, nn_module_stack
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
        ) = make_fake_inputs(mod, args, kwargs, dynamic_shapes)

        fake_params_buffers = make_fake_params_buffers(
            fake_mode, _get_params_buffers(mod)
        )
        with fake_mode:
            ep_non_strict = _export_non_strict(
                mod,
                fake_args,
                fake_kwargs,
                fake_params_buffers,
                constant_attrs,
                pre_dispatch=pre_dispatch,
                transform=_tuplify_outputs,
            )
        ep_non_strict.gm.meta["inline_constraints"] = {
            k: v
            for k, v in fake_mode.shape_env.var_to_range.items()
            if free_unbacked_symbols(k)
        }
        try:
            range_constraints = make_constraints(
                fake_mode,
                equalities_inputs,
                dynamic_shapes if dynamic_shapes else [],
                ep_non_strict.sig.input_specs,
                original_signature,
                ep_non_strict.gm,
            )
        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e))  # noqa: TRY200

        assert out_spec is not None

        gm = ep_non_strict.gm

        module_call_signatures = {
            strip_root(fqn): ModuleCallSignature(inputs=[], outputs=[], **specs)
            for fqn, specs in module_call_specs.items()
        }

        if len(preserve_module_call_signature) > 0:
            for node in gm.graph.nodes:
                if node.target == torch.ops.higher_order._export_tracepoint:
                    if "path" in node.kwargs:
                        path = strip_root(node.kwargs["path"])
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

            res = CollectTracepointsPass(module_call_signatures, ep_non_strict.sig)(gm)
            assert res is not None
            gm = res.graph_module

        _rewrite_non_persistent_buffers(mod, ep_non_strict.sig, ep_non_strict.constants)
        _verify_nn_module_stack(gm)
        _verify_stack_trace(gm)
        _verify_placeholder_names(gm, ep_non_strict.sig)
        exported_program = ExportedProgram(
            root=gm,
            graph=gm.graph,
            graph_signature=ep_non_strict.sig,
            state_dict=mod.state_dict(keep_vars=True),
            range_constraints=range_constraints,
            module_call_graph=_make_module_call_graph(
                _EXPORT_MODULE_HIERARCHY, orig_in_spec, out_spec, module_call_signatures
            ),
            example_inputs=(args, kwargs),
            constants=ep_non_strict.constants,
        )
        insert_deferred_runtime_asserts(
            exported_program.graph_module,
            fake_mode.shape_env,
            f"non strict exported program: {first_call_function_nn_module_stack(exported_program.graph)}",
            export=True,
        )
        return exported_program

    gm_torch_level = _export_to_torch_ir(
        mod,
        args,
        kwargs,
        dynamic_shapes,
        preserve_module_call_signature=preserve_module_call_signature,
        restore_fqn=False,  # don't need to restore because we will do it later
        _log_export_usage=False,
    )

    # We detect the fake_mode by looking at gm_torch_level's placeholders, this is the fake_mode created in dynamo.
    (
        fake_args,
        fake_kwargs,
        fake_params_buffers,
        dynamo_fake_mode,
    ) = _convert_input_to_fake(gm_torch_level, args, kwargs)

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

    # When aot_export lifts the params, we lose metadata (e.g. source_fn_stack, stack_trace)
    # from the param nodes as they are treated as fresh inputs
    # Therefore, we manually extract them before calling into aot_export
    params_buffers_to_node_meta = {}
    for node in gm_torch_level.graph.nodes:
        target = node.target
        meta = node.meta
        if node.op == "call_module":
            submodule = getattr(gm_torch_level, target)
            if isinstance(submodule, torch.nn.Module):
                for name, _ in submodule.named_parameters(
                    recurse=True, remove_duplicate=False
                ):
                    params_buffers_to_node_meta[target + "." + name] = meta

                for name, _ in submodule.named_buffers(
                    recurse=True, remove_duplicate=False
                ):
                    params_buffers_to_node_meta[target + "." + name] = meta

        if node.op == "get_attr":
            submodule = getattr(gm_torch_level, target)
            if not isinstance(submodule, torch.fx.GraphModule):
                params_buffers_to_node_meta[target] = meta

        # If the call_function uses param as input, we also need to update params' meta
        # with this call_function node's meta.
        # This is basically the same flow as torch.fx.traceback.preserve_meta()
        if node.op == "call_function" and not isinstance(
            node.target, torch._ops.HigherOrderOperator
        ):
            for arg in node._input_nodes:
                if arg.op == "get_attr":
                    for entry in torch.fx.proxy._COPY_META_FIELDS:
                        if entry in meta:
                            params_buffers_to_node_meta[arg.target][entry] = meta[entry]

    # Fix the graph output signature to be tuple if scalar
    out_spec = orig_out_spec = gm_torch_level._out_spec
    assert out_spec is not None
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

    # NOTE: graph module expects only positional args
    ep_non_strict = _export_non_strict(
        gm_torch_level,
        _convert_to_positional_args(orig_arg_names, fake_args, fake_kwargs),
        {},
        fake_params_buffers,
        constant_attrs,
        pre_dispatch=pre_dispatch,
    )

    gm = ep_non_strict.gm
    export_graph_signature = ep_non_strict.sig
    constants = ep_non_strict.constants

    # Don't copy over nn_module_stack, stack_trace metadata for params/buffers nodes
    for metadata in params_buffers_to_node_meta.values():
        metadata.pop("nn_module_stack", None)
        metadata.pop("stack_trace", None)

    # After aot_export, set the param/buffer metadata back into placeholders
    # Technically, users can still construct this data from param names
    # without relying on this metadata
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.target in export_graph_signature.inputs_to_parameters:
                param_name = export_graph_signature.inputs_to_parameters[node.target]
                if param_name in params_buffers_to_node_meta:
                    for k, v in params_buffers_to_node_meta[param_name].items():
                        node.meta[k] = v
            if node.target in export_graph_signature.inputs_to_buffers:
                buffer_name = export_graph_signature.inputs_to_buffers[node.target]
                if buffer_name in params_buffers_to_node_meta:
                    for k, v in params_buffers_to_node_meta[buffer_name].items():
                        node.meta[k] = v

    # The unbacked symint symbols are updated in aot_export
    # so we serialize them here instead of inside dynamo

    gm.meta["inline_constraints"] = {
        k: v
        for k, v in dynamo_fake_mode.shape_env.var_to_range.items()
        if free_unbacked_symbols(k)
    }

    num_lifted = next(
        (
            i
            for i, s in enumerate(export_graph_signature.input_specs)
            if s.kind == InputKind.USER_INPUT
        ),
        len(export_graph_signature.input_specs),
    )
    range_constraints = _process_constraints(
        dynamo_fake_mode,
        gm,
        num_lifted,
        flat_args,
    )

    # Do some cleanups on the graph module to restore the state dict to the
    # expected form. Each of these steps should probably get fixed upstream.
    # 1. Remove tensor constants that were added as buffers.
    _rewrite_dynamo_tensor_constants(
        orig_mod_buffers=set(mod.buffers()),
        traced_mod_buffers=dict(gm_torch_level.named_buffers()),
        graph_signature=ep_non_strict.sig,
        constants=ep_non_strict.constants,
    )
    # 2. Restore FQN of param/buffers
    param_buffer_table: Dict[str, str] = _get_param_buffer_mapping(mod, gm_torch_level)
    _replace_param_buffer_names(param_buffer_table, export_graph_signature)

    # 3. Remove non-persistent buffers from the graph signature
    _rewrite_non_persistent_buffers(mod, ep_non_strict.sig, ep_non_strict.constants)

    # 4. Rewrite constants to have the same FQN as the original module.
    _remap_constants(constant_attrs, export_graph_signature, constants)

    # 5. Rename constants nodes in graph module from buffers to constants
    _rename_constants_nodes(gm, export_graph_signature)

    module_call_signatures = {
        fqn: ModuleCallSignature(inputs=[], outputs=[], **specs)
        for fqn, specs in gm_torch_level.meta["module_call_specs"].items()
    }

    if len(preserve_module_call_signature) > 0:
        res = CollectTracepointsPass(module_call_signatures, export_graph_signature)(gm)
        assert res is not None
        gm = res.graph_module

    if len(range_constraints) > 0:
        res = _AddRuntimeAssertionsForInlineConstraintsPass(range_constraints)(gm)
        assert res is not None
        gm = res.graph_module

    assert orig_out_spec is not None
    _verify_nn_module_stack(gm)
    _verify_stack_trace(gm)
    _verify_placeholder_names(gm, export_graph_signature)
    exported_program = ExportedProgram(
        root=gm,
        graph=gm.graph,
        graph_signature=export_graph_signature,
        state_dict=mod.state_dict(keep_vars=True),
        range_constraints=range_constraints,
        module_call_graph=_make_module_call_graph(
            _EXPORT_MODULE_HIERARCHY,
            orig_in_spec,
            orig_out_spec,
            module_call_signatures,
        ),
        example_inputs=(args, kwargs),
        constants=constants,
    )
    log.debug("Exported program from AOTAutograd:\n%s", exported_program)

    return exported_program
