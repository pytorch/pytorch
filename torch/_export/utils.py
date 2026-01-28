# mypy: allow-untyped-defs
import ast
import copy
import dataclasses
import functools
import inspect
import json
import math
import operator
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from inspect import ismethod, Parameter
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx._utils import first_call_function_nn_module_stack
from torch.fx.experimental.proxy_tensor import PreDispatchTorchFunctionMode
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts


if TYPE_CHECKING:
    import sympy

    from torch._export.passes.lift_constants_pass import ConstantAttrMap
    from torch._ops import OperatorBase
    from torch.export import ExportedProgram
    from torch.export.graph_signature import ExportGraphSignature

from torch.export.graph_signature import CustomObjArgument, InputKind, OutputKind
from torch.fx._pytree import (
    _deregister_pytree_flatten_spec,
    register_pytree_flatten_spec,
)
from torch.utils._pytree import (
    _deregister_pytree_node,
    _register_pytree_node,
    Context,
    FlattenFunc,
    FromDumpableContextFn,
    GetAttrKey,
    KeyPath,
    keystr,
    MappingKey,
    SequenceKey,
    ToDumpableContextFn,
    tree_flatten_with_path,
    UnflattenFunc,
)


placeholder_prefixes = {
    InputKind.USER_INPUT: "",
    InputKind.PARAMETER: "p_",
    InputKind.BUFFER: "b_",
    InputKind.CONSTANT_TENSOR: "c_",
    InputKind.CUSTOM_OBJ: "obj_",
    InputKind.TOKEN: "token",
}

_DISABLE_ATEN_TO_ASSERTION_PASS = False


def _collect_and_set_constant_attrs(
    graph_signature, constants, mod
) -> "ConstantAttrMap":
    # the exported module will store constants & non-persistent buffers such that
    # retracing treats them as persistent buffers, so we inform the constants lifting pass
    # and overwrite the new graph signature using the previous program. This is intended to only be used
    # in run_decompositions where we still have access to original EP.
    from torch._export.passes.lift_constants_pass import ConstantAttrMap

    constant_attrs = ConstantAttrMap()
    non_persistent_buffers = {
        spec.target
        for spec in graph_signature.input_specs
        if spec.kind == InputKind.BUFFER and not spec.persistent
    }
    for name, value in constants.items():
        if name in non_persistent_buffers:
            continue
        # recursive getattr
        _mod = mod
        *atoms, attr = name.split(".")
        for atom in atoms:
            _mod = getattr(_mod, atom)
        # remove as buffer, reassign as constant/non-persistent buffer
        _mod._buffers.pop(attr, None)
        setattr(_mod, attr, value)
        constant_attrs.add(value, name)
    return constant_attrs


def _register_constants_as_buffers(
    mod: torch.fx.GraphModule, state_dict, non_persistent_buffers
):
    # TODO some annoying circular dependency issue
    from torch.export.unflatten import _assign_attr, _AttrKind

    temp_registered_constants = set()

    for node in mod.graph.nodes:
        if node.op == "get_attr":
            target = torch.fx.graph_module._get_attr(mod, node.target)
            if isinstance(target, torch.Tensor):
                # Make sure we also check if the original buffer is
                # non persistent as well.
                if (node.target not in state_dict) and (
                    node.target not in non_persistent_buffers
                ):
                    torch.fx.graph_module._del_attr(mod, node.target)
                    _assign_attr(target, mod, node.target, _AttrKind.BUFFER, False)
                    temp_registered_constants.add(node.target)

    mod.recompile()

    return temp_registered_constants


def _override_graph_signature_for_temp_registered_constants(
    sig: "ExportGraphSignature", temp_registered_constants
):
    for spec in sig.input_specs:
        if spec.target in temp_registered_constants:
            spec.kind = InputKind.CONSTANT_TENSOR
            spec.persistent = None

    for spec in sig.output_specs:
        if (
            spec.kind == OutputKind.BUFFER_MUTATION
            and spec.target in temp_registered_constants
        ):
            raise RuntimeError(
                f"Constant {spec.target} is mutated in the forward method. Pls register it as buffer"
            )

    return sig


def _overwrite_signature_for_non_persistent_buffers(
    old_sig: "ExportGraphSignature", new_sig: "ExportGraphSignature"
):
    # overwrite signature for non-persistent buffers
    non_persistent_buffers = {
        spec.target
        for spec in old_sig.input_specs
        if spec.kind == InputKind.BUFFER and not spec.persistent
    }

    for spec in new_sig.input_specs:
        if spec.kind == InputKind.BUFFER and spec.target in non_persistent_buffers:
            spec.persistent = False
    return new_sig


def _collect_param_buffer_metadata(mod: torch.fx.GraphModule) -> dict[str, Any]:
    """
    Param/buffer metadata needs to be saved before lowering to aten IR
    because aten IR lifts them, as a result, automatic preservation doesn't work.
    This is intended to be called on the strict mode tracing right before lowering to
    aten IR OR run_decomposition pass.
    """
    params_buffers_to_node_meta = {}

    def _getattr(model: torch.fx.GraphModule, attr_name: str):
        *prefix, field = attr_name.split(".")
        t = model
        for item in prefix:
            t = getattr(t, item, None)  # type: ignore[assignment]
            assert t is not None

        return getattr(t, field)

    for node in mod.graph.nodes:
        target = node.target
        meta = node.meta
        if node.op == "call_module":
            submodule = _getattr(mod, target)
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
            submodule = _getattr(mod, target)
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
                        #  the custom field should not be copied
                        if entry == "custom":
                            continue
                        if entry in meta:
                            params_buffers_to_node_meta[arg.target][entry] = meta[entry]

    return params_buffers_to_node_meta


def _maybe_find_pre_dispatch_tf_mode_for_export():
    if not torch._C._is_torch_function_mode_enabled():
        return None

    torch_function_mode_stack = torch.overrides._get_current_function_mode_stack()

    pre_dispatch_tf_modes = [
        mode
        for mode in torch_function_mode_stack
        if isinstance(mode, PreDispatchTorchFunctionMode)
    ]

    assert len(pre_dispatch_tf_modes) <= 1, (
        f"Expected only one PreDispatchTorchFunctionMode, found {len(pre_dispatch_tf_modes)}"
    )

    if len(pre_dispatch_tf_modes) == 0:
        return None

    mode = pre_dispatch_tf_modes[0]
    return mode


def _populate_param_buffer_metadata_to_new_gm(
    params_buffers_to_node_meta: dict[str, Any],
    gm: torch.fx.GraphModule,
    new_sig: "ExportGraphSignature",
) -> None:
    """
    Given that we collected param'buffer metadata before, we put them back in
    newly traced graph module
    """
    # Don't copy over nn_module_stack, stack_trace metadata for params/buffers nodes
    for metadata in params_buffers_to_node_meta.values():
        metadata.pop("nn_module_stack", None)
        metadata.pop("stack_trace", None)

    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.target in new_sig.inputs_to_parameters:
                param_name = new_sig.inputs_to_parameters[node.target]
                if param_name in params_buffers_to_node_meta:
                    for k, v in params_buffers_to_node_meta[param_name].items():
                        node.meta[k] = v
            if node.target in new_sig.inputs_to_buffers:
                buffer_name = new_sig.inputs_to_buffers[node.target]
                if buffer_name in params_buffers_to_node_meta:
                    for k, v in params_buffers_to_node_meta[buffer_name].items():
                        node.meta[k] = v


def _get_shape_env_from_gm(gm: torch.fx.GraphModule):
    vals = [
        node.meta["val"]
        for node in gm.graph.nodes
        if node.meta.get("val", None) is not None
    ]

    fake_mode = _detect_fake_mode_from_gm(gm)
    if fake_mode is not None:
        return fake_mode.shape_env
    for v in vals:
        if isinstance(v, torch.SymInt):
            return v.node.shape_env


def _rename_without_collisions(
    name_map: dict[str, str],
    find_available: dict[str, int],
    used_names: set[str],
    orig_name: str,
    name: str,
    is_placeholder: bool = False,
):
    """
    Renames nodes to avoid name collisions, with suffixing.
    name_map: map from original name to new name
    find_available: map prefix to available suffix
    used_names: cache of used names
    orig_name: mapping key
    name: candidate name (potentially suffixed, e.g. mul_2)
    is_placeholder: if the node is a placeholder, avoid detecting suffix
    """
    match = re.match(r"(.*)_(\d+)", name)
    key = name

    if match and not is_placeholder:
        prefix, n = match.group(1), match.group(2)
        key = prefix

    new_name = name
    if new_name in used_names:
        new_name = f"{key}_{find_available[key] + 1}"

    match = re.match(r"(.*)_(\d+)", new_name)
    if match:
        prefix, n = match.group(1), match.group(2)
        if int(n) > find_available[prefix]:
            find_available[prefix] = int(n)

    name_map[orig_name] = new_name
    used_names.add(new_name)

    return name_map[orig_name]


def get_keystr(key_path: KeyPath) -> str:
    """For a given index into the flat_args, return a human readable string
    describing how to access it, e.g. "*args["foo"][0].bar"
    """
    # Prefix the keypath with "*args" or "**kwargs" to make it clearer where
    # the arguments come from. Ultimately we ought to serialize the
    # original arg names for the best error message here.
    args_kwargs_key_path = key_path[0]
    assert isinstance(args_kwargs_key_path, SequenceKey)
    if args_kwargs_key_path.idx == 0:
        return f"*args{keystr(key_path[1:])}"
    else:
        kwarg_key = key_path[1]
        assert isinstance(kwarg_key, (GetAttrKey, MappingKey))
        name = str(kwarg_key)[1:-1]  # get rid of the enclosed []
        return f"{name}{keystr(key_path[2:])}"


def _check_symint(
    symint: Union[int, torch.SymInt],
    arg: int,
    range_constraints,
    unification_map,
    keypath: KeyPath,
    i: Optional[int] = None,
) -> None:
    from torch.export.dynamic_shapes import _IntWrapper

    if (
        isinstance(arg, torch.SymInt)
        and not arg.node.expr.is_number
        or isinstance(arg, _IntWrapper)
    ):
        # This can happen when, say, arg is a fake tensor.
        # We do not run checks on symbolic shapes of fake inputs as
        # such checks can affect the shape env.
        return

    import sympy

    from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
        _convert_range_to_int,
    )
    from torch.utils._sympy.solve import try_solve

    if isinstance(symint, torch.SymInt) and len(symint.node.expr.free_symbols) == 1:
        symbol = next(iter(symint.node.expr.free_symbols))
        if symbol in unification_map:
            existing_dim = symint.node.expr.subs(unification_map)
            if arg != existing_dim:
                path = get_keystr(keypath)
                if i is not None:
                    path += f".shape[{i}]"
                raise RuntimeError(
                    f"Expected input at {path} to be equal to {existing_dim}, but got {arg}",
                )
        else:
            if isinstance(symint.node.expr, sympy.Symbol):
                # Short cut for try_solve below. Also useful in cases where
                # sympy.Eq(symint.node.expr, arg) would evaluate to False
                # purely because symbol is constrained to be size-like,
                # e.g., when symint.node.expr = symbol and arg = 0.
                unification_map[symbol] = int(arg)
            else:
                solution = try_solve(sympy.Eq(symint.node.expr, arg), symbol)
                if solution is None:
                    path = get_keystr(keypath)
                    if i is not None:
                        path += f".shape[{i}]"
                    raise RuntimeError(  # noqa: B904
                        f"Expected input {path} = {arg} to be "
                        f"of the form {symint.node.expr}, where {symbol} is an integer"
                    )
                else:
                    unification_map[symbol] = int(solution[1])

        if symint.node.expr in range_constraints:
            min_val, max_val = _convert_range_to_int(
                range_constraints[symint.node.expr]
            )
            # NOTE: we allow dimensions to be 0/1 at runtime
            if min_val > 2:
                if arg < min_val:
                    path = get_keystr(keypath)
                    if i is not None:
                        path += f".shape[{i}]"
                    raise RuntimeError(
                        f"Expected input at {path} to be >= {min_val}, but got {arg}",
                    )
            if max_val < math.inf:
                if arg > max_val:
                    path = get_keystr(keypath)
                    if i is not None:
                        path += f".shape[{i}]"
                    raise RuntimeError(
                        f"Expected input at {path} to be <= {max_val}, but got {arg}",
                    )
    elif isinstance(symint, torch.SymInt) and not symint.node.expr.is_number:
        # this means we deferred a guard from export analysis to runtime, let this pass
        # we'll add a runtime assert checking equality to this replacement expression
        pass
    elif arg != int(symint):
        path = get_keystr(keypath)
        if i is not None:
            path += f".shape[{i}]"
        raise RuntimeError(
            f"Expected input at {path} to be equal to {symint}, but got {arg}. "
            "If you meant for this dimension to be dynamic, please re-export and specify dynamic_shapes "
            "(e.g. with Dim.DYNAMIC)"
        )


def _check_input_constraints_for_graph(
    input_placeholders: list[torch.fx.Node], flat_args_with_path, range_constraints
) -> None:
    if len(flat_args_with_path) != len(input_placeholders):
        raise RuntimeError(
            "Unexpected number of inputs "
            f"(expected {len(input_placeholders)}, got {len(flat_args_with_path)})"
        )
    # NOTE: export already guarantees that the same symbol is used in metadata
    # for all InputDims related by equality constraints, so we can just unify
    # symbols with given input dimension values to check equality constraints.
    unification_map: dict[sympy.Symbol, Any] = {}
    for (key_path, arg), node in zip(flat_args_with_path, input_placeholders):
        node_val = node.meta.get("val")
        if isinstance(node_val, FakeTensor):
            if not isinstance(arg, torch.Tensor):
                raise RuntimeError(
                    f"Expected input at {get_keystr(key_path)} to be a tensor, but got {type(arg)}",
                )

            if len(node_val.shape) != len(arg.shape):
                raise RuntimeError(
                    f"Unexpected number of dimensions in input at {get_keystr(key_path)}.shape "
                    f"(expected {node_val.shape}, got {arg.shape})"
                )

            for j, (arg_dim, node_dim) in enumerate(zip(arg.shape, node_val.shape)):
                _check_symint(
                    node_dim, arg_dim, range_constraints, unification_map, key_path, j
                )

        elif isinstance(node_val, (int, float, str)):
            if type(arg) is not type(node_val) or arg != node_val:
                raise RuntimeError(
                    f"Expected input at {get_keystr(key_path)} to be equal to {node_val}, but got {arg}",
                )
        elif isinstance(node_val, torch.SymInt):
            _check_symint(
                node_val,
                arg,
                range_constraints,
                unification_map,
                key_path,
                None,
            )


def register_dataclass_as_pytree_node(
    cls: type[Any],
    flatten_fn: Optional[FlattenFunc] = None,
    unflatten_fn: Optional[UnflattenFunc] = None,
    *,
    serialized_type_name: Optional[str] = None,
    to_dumpable_context: Optional[ToDumpableContextFn] = None,
    from_dumpable_context: Optional[FromDumpableContextFn] = None,
    return_none_fields: bool = False,
) -> None:
    assert dataclasses.is_dataclass(cls), (
        f"Only dataclasses can be registered with this function: {cls}"
    )

    @torch._dynamo.dont_skip_tracing
    def default_flatten_fn(obj: Any) -> tuple[list[Any], Context]:
        flattened = []
        flat_names = []
        none_names = []
        for f in dataclasses.fields(obj):
            name, val = f.name, getattr(obj, f.name)
            if val is not None or return_none_fields:
                flattened.append(val)
                flat_names.append(name)
            else:
                none_names.append(name)
        return flattened, [flat_names, none_names]

    @torch._dynamo.dont_skip_tracing
    def default_unflatten_fn(values: Iterable[Any], context: Context) -> Any:
        flat_names, none_names = context
        return cls(**dict(zip(flat_names, values)), **dict.fromkeys(none_names))

    @torch._dynamo.dont_skip_tracing
    def default_flatten_fn_with_keys(obj: Any) -> tuple[list[Any], Context]:
        flattened, (flat_names, _none_names) = flatten_fn(obj)  # type: ignore[misc]
        return [(MappingKey(k), v) for k, v in zip(flat_names, flattened)], flat_names

    flatten_fn = flatten_fn if flatten_fn is not None else default_flatten_fn
    unflatten_fn = unflatten_fn if unflatten_fn is not None else default_unflatten_fn

    if (to_dumpable_context is None) ^ (from_dumpable_context is None):
        raise ValueError(
            f"Both to_dumpable_context and from_dumpable_context for {cls} must "
            "be None or registered."
        )

    _register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        flatten_with_keys_fn=default_flatten_fn_with_keys,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
    )


def is_param(program: "ExportedProgram", node: torch.fx.Node) -> bool:
    """
    Checks if the given node is a parameter within the exported program
    """

    return node.name in program.graph_signature.inputs_to_parameters


def get_param(
    program: "ExportedProgram",
    node: torch.fx.Node,
) -> Optional[torch.nn.Parameter]:
    """
    Returns the parameter associated with the given node in the exported program.
    Returns None if the node is not a parameter within the exported program
    """

    if is_param(program, node):
        parameter_name = program.graph_signature.inputs_to_parameters[node.name]
        return program.state_dict[parameter_name]

    return None


def is_buffer(program: "ExportedProgram", node: torch.fx.Node) -> bool:
    """
    Checks if the given node is a buffer within the exported program
    """

    return node.name in program.graph_signature.inputs_to_buffers


def get_buffer(
    program: "ExportedProgram",
    node: torch.fx.Node,
) -> Optional[torch.Tensor]:
    """
    Returns the buffer associated with the given node in the exported program.
    Returns None if the node is not a buffer within the exported program
    """

    if is_buffer(program, node):
        buffer_name = program.graph_signature.inputs_to_buffers[node.name]
        if buffer_name in program.graph_signature.non_persistent_buffers:
            return program.constants[buffer_name]
        else:
            return program.state_dict[buffer_name]

    return None


def is_lifted_tensor_constant(
    program: "ExportedProgram",
    node: torch.fx.Node,
) -> bool:
    """
    Checks if the given node is a lifted tensor constant within the exported program
    """

    return node.name in program.graph_signature.inputs_to_lifted_tensor_constants


def get_lifted_tensor_constant(
    program: "ExportedProgram",
    node: torch.fx.Node,
) -> Optional[torch.Tensor]:
    """
    Returns the lifted tensor constant associated with the given node in the exported program.
    Returns None if the node is not a lifted tensor constant within the exported program
    """

    if is_lifted_tensor_constant(program, node):
        lifted_tensor_name = program.graph_signature.inputs_to_lifted_tensor_constants[
            node.name
        ]
        return program.constants[lifted_tensor_name]

    return None


def sequential_split(
    gm: torch.fx.GraphModule,
    node_call_back: Callable[[torch.fx.Node], Union[torch.fx.Node, bool]],
) -> torch.fx.GraphModule:
    """
    sequential_split creates a new graph module that splits the input graph module into multiple submodules
    based on the node_call_back. It doesn't mutate the input graph module. The node_call_back should return
    True if the node is a delimiter.  Delimiter will be the first node in the next submodule.
    """
    from torch.fx.passes.split_module import split_module

    split_map = {}
    split_id = 0
    for node in gm.graph.nodes:
        if node_call_back(node):
            split_id += 1
        split_map[node] = split_id

    new_gm = split_module(
        gm,
        gm,
        lambda node: split_map[node],
        keep_original_order=True,
        keep_original_node_name=True,
    )
    # Keep the codegen from original graph module to preserve e.g. pytree info.
    new_gm.graph._codegen = gm.graph._codegen
    new_gm.recompile()
    return new_gm


def nodes_filter(nodes: list[torch.fx.Node], node_call_back) -> list[torch.fx.Node]:
    """Returns the nodes that match the node_call_back as a list."""
    return [node for node in nodes if node_call_back(node)]


@contextmanager
def _disable_aten_to_metadata_assertions():
    global _DISABLE_ATEN_TO_ASSERTION_PASS
    orig_val = _DISABLE_ATEN_TO_ASSERTION_PASS
    _DISABLE_ATEN_TO_ASSERTION_PASS = True
    try:
        yield
    finally:
        _DISABLE_ATEN_TO_ASSERTION_PASS = orig_val


def _insert_aten_to_metadata_assert_pass(gm: torch.fx.GraphModule) -> None:
    from torch._export.passes._node_metadata_hook import (
        _node_metadata_hook,
        _set_node_metadata_hook,
    )

    if _DISABLE_ATEN_TO_ASSERTION_PASS:
        return

    aten_to_variants = [
        torch.ops.aten.to.device,
        torch.ops.aten.to.dtype,
        torch.ops.aten.to.dtype_layout,
    ]
    for node in gm.graph.nodes:
        if node.target in aten_to_variants:
            if (
                node.prev.target is torch.ops.aten._assert_tensor_metadata.default
                and node.args[0] == node.prev.args[0]
            ):
                # skip if already guarded
                continue

            if (tensor_val := node.args[0].meta.get("val")) is not None:
                with (
                    gm.graph.inserting_before(node),
                    _set_node_metadata_hook(
                        gm,
                        functools.partial(
                            _node_metadata_hook,
                            metadata={
                                "stack_trace": node.meta.get("stack_trace"),
                                "nn_module_stack": node.meta.get("nn_module_stack"),
                            },
                        ),
                    ),
                ):
                    gm.graph.call_function(
                        torch.ops.aten._assert_tensor_metadata.default,
                        args=(node.args[0],),
                        kwargs={
                            "dtype": tensor_val.dtype,
                            "device": tensor_val.device,
                            "layout": tensor_val.layout,
                        },
                    )


def apply_runtime_assertion_pass(gm: torch.fx.GraphModule, graph_signature):
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
        with _set_node_metadata_hook(
            gm,
            functools.partial(
                _node_metadata_hook, metadata={"stack_trace": stack_trace}
            ),
        ):
            shape_env = _get_shape_env_from_gm(gm)
            if shape_env:
                insert_deferred_runtime_asserts(
                    gm,
                    shape_env,
                    f"exported program: {first_call_function_nn_module_stack(gm.graph)}",
                    export=True,
                )

        # insert runtime assertions for aten.to nodes
        _insert_aten_to_metadata_assert_pass(gm)

    # update output specs
    gm.recompile()
    graph_signature.user_outputs = _graph_output_names(gm)
    return gm, graph_signature


def nodes_first(
    nodes: list[torch.fx.Node], node_call_back=None
) -> Optional[torch.fx.Node]:
    """
    Returns the first node that matches the node_call_back. If no node matches, returns None.
    When node_call_back is None, returns the first node in the node list.
    """
    ret = nodes_filter(nodes, node_call_back if node_call_back else lambda node: True)
    if len(ret) > 0:
        return ret[0]
    return None


def nodes_count(nodes: list[torch.fx.Node], node_call_back) -> int:
    """Returns the number of nodes that match the node_call_back."""
    return len(nodes_filter(nodes, node_call_back))


def nodes_map(nodes: list[torch.fx.Node], node_call_back) -> list[torch.fx.Node]:
    """
    Sequentially visit the nodes list and invoke node_call_back on each element.
    Returns the nodes list after the node_call_back is invoked on each element.
    """
    for node in nodes:
        node_call_back(node)
    return nodes


def node_replace_(old_node: torch.fx.Node, new_node: torch.fx.Node) -> None:
    """
    Replace all uses of old_node with new_node.
    """
    old_node.replace_all_uses_with(new_node)
    old_node.users.clear()
    old_node.graph.erase_node(old_node)


def _update_gm_meta_if_possible(gm: torch.fx.GraphModule, mod: torch.nn.Module) -> None:
    if (
        isinstance(mod, torch.fx.GraphModule)
        and hasattr(mod, "meta")
        and "custom" in mod.meta
    ):
        gm.meta.update({"custom": mod.meta["custom"]})


def node_inline_(call_mod_node: torch.fx.Node) -> Optional[torch.fx.GraphModule]:
    """
    Inline the submodule of the given node into the parent module.
    Note: we only support the case where submodule takes tensors inputs.
    """
    assert call_mod_node.op == "call_module"
    gm = call_mod_node.graph.owning_module
    assert gm is not None

    assert isinstance(call_mod_node.target, str)
    sub_gm = getattr(gm, call_mod_node.target)

    phs = (node for node in sub_gm.graph.nodes if node.op == "placeholder")
    body = (
        node for node in sub_gm.graph.nodes if node.op not in ("placeholder", "output")
    )
    output = [node for node in sub_gm.graph.nodes if node.op == "output"]

    for ph, arg in zip(phs, call_mod_node.args):
        assert isinstance(arg, torch.fx.Node)
        node_replace_(ph, arg)

    with gm.graph.inserting_before(call_mod_node):
        for node in body:
            new_node = gm.graph.node_copy(node)
            if node.op == "get_attr":
                new_target_name = new_node.target
                if hasattr(gm, new_target_name):
                    # Loop through and find the "submod_{i}" that have no name collision
                    i = 1
                    new_target_name = f"submod_{i}"
                    while hasattr(gm, new_target_name):
                        i += 1
                        new_target_name = f"submod_{i}"
                new_node.target = new_target_name
                setattr(gm, new_node.target, getattr(sub_gm, node.target))
            node_replace_(node, new_node)

        if len(output) > 0:
            assert len(output) == 1 and len(output[0].args) == 1
            new_output = output[0].args[0]

            if isinstance(new_output, torch.fx.Node):
                # Clear the users of the output node and set
                # the users to be the users of original call_module node.
                new_output.users.clear()
                node_replace_(call_mod_node, new_output)
            elif isinstance(new_output, (list, tuple)):
                # Pop subgraph output node from users.
                for node in new_output:
                    node.users.pop(output[0])

                # Inline the get_item calls for the output node.
                get_item_users = nodes_filter(
                    list(call_mod_node.users.keys()),
                    lambda node: node.op == "call_function"
                    and node.target is operator.getitem,
                )
                # get_item_node.args[1] is the idx referring to new_output[idx]
                nodes_map(
                    get_item_users,
                    lambda get_item_node: node_replace_(
                        get_item_node,
                        new_output[get_item_node.args[1]],
                    ),
                )
                call_mod_node.graph.erase_node(call_mod_node)
            else:
                raise NotImplementedError(
                    f"Unsupported output type {type(new_output)}. Expect it to be a Node or a list/tuple of Nodes."
                )
        else:
            call_mod_node.graph.erase_node(call_mod_node)

    gm.delete_all_unused_submodules()
    gm.recompile()
    return gm


def _get_torch_jit_trace_forward_signature(mod: torch.nn.Module) -> inspect.Signature:
    """
    Get source code and parse argument names using AST. The function returns
    a signature of the forward() function.

    # TODO: Directly provide inspect.signature compatible TS-d module.
    """
    ast_mod = ast.parse(mod.code)  # type: ignore[call-overload]
    ast_func_def: ast.FunctionDef = ast_mod.body[0]

    # FIXME(jiashenc): TorchScript should only allow positional or keywords arguments.
    arg_type_map = {"args": Parameter.POSITIONAL_OR_KEYWORD}

    # Traverse all argument types in AST tree and create associated parameters.
    param_list = []
    for arg_type, param_type in arg_type_map.items():
        arg_name_list = [a.arg for a in getattr(ast_func_def.args, arg_type)]
        for arg_name in arg_name_list:
            if arg_name == "self":
                continue  # Skip self argument.
            param_list.append(inspect.Parameter(arg_name, param_type))

    return inspect.Signature(parameters=param_list)


def _bind_signature_to_inputs(mod, fake_args, fake_kwargs):
    if isinstance(mod, (torch.jit.ScriptModule, torch.jit.TracedModule)):
        sig = _get_torch_jit_trace_forward_signature(mod)

        # Sanity check for placeholder names coming from TorchScript.
        assert len(sig.parameters) == len(fake_args) + len(fake_kwargs), (
            "Arguments other than POSITIONAL_OR_KEYWORD kinds in forward() "
            "are not supported in _get_torch_jit_trace_forward_signature"
        )
    else:
        sig = inspect.signature(mod.forward)

    # Rather than binding both fake_args and fake_kwargs to sig names, we
    # (partially) bind only fake_args, while reusing fake_kwarg names. This
    # ensures that fake_kwargs do not get reordered, which is important to
    # match flattened user inputs.
    return {**sig.bind_partial(*fake_args).arguments, **fake_kwargs}


def _build_cache(name, find_available, used_names):
    used_names.add(name)
    match = re.match(r"(.*)_(\d+)", name)
    if match:
        prefix, n = match.group(1), match.group(2)
        if int(n) > find_available[prefix]:
            find_available[prefix] = int(n)


def _name_hoo_subgraph_placeholders(gm: torch.fx.GraphModule) -> None:
    """
    Propagate placeholder names from the top-level graph into HigherOrderOp subgraphs,
    and handle collisions with non-placeholders by count suffixing.
    Different HOO subgraph types have different input schemas, so we first enumerate them
    and gather the top-level named placeholder nodes.
    """

    # gather all HOO subgraphs and their top-level named placeholder nodes
    subgraph_ph_tuples: list[tuple[torch.fx.GraphModule, list[torch.fx.Node]]] = []
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
        name_map: dict[str, str] = {}
        find_available: dict[str, int] = defaultdict(int)
        used_names: set[str] = set()
        for i, node in enumerate(subgraph.graph.nodes):
            if i < len(hoo_phs):  # placeholder, retain name
                name_map[node.name] = hoo_phs[i].name
                node.name = node.target = hoo_phs[i].name
                _build_cache(node.name, find_available, used_names)
            else:  # non-placeholder, check for collisions
                node.name = _rename_without_collisions(
                    name_map, find_available, used_names, node.name, node.name
                )

        # recurse and recompile
        _name_hoo_subgraph_placeholders(subgraph)
        subgraph.recompile()


def _assign_new_node_names(
    gm: torch.fx.GraphModule,
    name_map: dict[str, str],
    custom_meta: dict[str, Any],
) -> None:
    """
    Assign new names to all nodes, in the graph module, from name map.
    """
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            assert node.name in name_map
            node.name = node.target = name_map[node.name]
            if node.name in custom_meta:
                if node.meta.get("custom") is None:
                    node.meta["custom"] = {}
                else:
                    # Assert if any existing key has different value
                    for k, v in node.meta["custom"].items():
                        if (
                            k in custom_meta[node.name]
                            and v != custom_meta[node.name][k]
                        ):
                            raise AssertionError(
                                f"Mismatch in custom metadata for key {k}. Value in "
                                f"node.meta is {v} and value in custom_meta is {custom_meta[node.name][k]}."
                            )
                node.meta["custom"].update(custom_meta[node.name])
            # if the constant obj is an input, we also need to update meta["val"]
            # because this is created before the placeholder naming pass
            if isinstance(node.meta["val"], CustomObjArgument):
                node.meta["val"].name = node.name
        elif node.name in name_map:
            node.name = name_map[node.name]


def placeholder_naming_pass(
    gm: torch.fx.GraphModule,
    export_graph_signature: "ExportGraphSignature",
    mod: torch.nn.Module,
    fake_args,
    fake_kwargs,
    fake_params_buffers,
    constants: dict[str, Any],
) -> None:
    """
    This pass is run at the end of _export_non_strict() to assign better placeholder node names:
        - User inputs:
            These follow the signature of mod.forward(), e.g. forward(x, y) produces nodes x, y.
            For nested inputs from dictionaries, lists, tuples, or dataclasses,
            the names are a concatenation of the path to the tensor.
                e.g. x = {
                    'a': torch.randn(),
                    'b': [torch.randn(), torch.randn()]
                }
            produces nodes x_a, x_b_0, x_b_1.
        - Parameters/buffers/constants/custom objects:
            These follow the FQN of the object, prefixed by "p", "b", "c", "obj" respectively.
                e.g. self.bar.l0.weight produces "p_bar_l0_weight".
        - Effect tokens:
            These are named token, token_1, ...
    """

    custom_meta: dict[str, Any] = {}
    if isinstance(mod, torch.fx.GraphModule):
        for node in mod.graph.nodes:
            if "custom" in node.meta:
                custom_meta[node.name] = node.meta["custom"]

    def _strip_name(x):
        if x.startswith("L__self___"):
            x = x[len("L__self___") :]
        elif x.startswith("self_"):
            x = x[len("self_") :]
        x = re.sub(r"[^a-zA-Z0-9]", "_", x)
        return x

    def _extract_pytree_key(x):
        if isinstance(x, MappingKey):
            x = re.sub(r"[^a-zA-Z0-9]", "_", str(x.key))
            return x
        elif isinstance(x, SequenceKey):
            return str(x.idx)
        elif isinstance(x, GetAttrKey):
            return x.name
        else:
            raise RuntimeError(f"Pytree key of type {type(x)} not handled for {x}")

    name_map: dict[str, str] = {}
    find_available: dict[str, int] = defaultdict(int)
    used_names: set[str] = set()

    # map user input names with mod.forward() signature
    combined_args = _bind_signature_to_inputs(mod, fake_args, fake_kwargs)

    flat_args_with_path, _ = tree_flatten_with_path(combined_args)
    user_input_names = [
        spec.arg.name
        for spec in export_graph_signature.input_specs
        if spec.kind == InputKind.USER_INPUT
    ]

    # use pytree path to name nested user inputs
    for (arg_path, _arg), user_input_name in zip(flat_args_with_path, user_input_names):
        if user_input_name:
            _rename_without_collisions(
                name_map,
                find_available,
                used_names,
                user_input_name,
                placeholder_prefixes[InputKind.USER_INPUT]
                + "_".join(_extract_pytree_key(x).lower() for x in arg_path),
                is_placeholder=True,
            )

    # use graph signature input specs to map param/buffer/constant names
    # name effect tokens as token, token_1, ... (these aren't visible to user)
    for spec in export_graph_signature.input_specs:
        if spec.kind == InputKind.USER_INPUT:
            continue
        if spec.kind == InputKind.TOKEN:
            base_name = ""
        else:
            base_name = _strip_name(spec.target).lower()
        base_name = re.sub(r"[^a-zA-Z0-9]", "_", base_name)

        _rename_without_collisions(
            name_map,
            find_available,
            used_names,
            spec.arg.name,
            placeholder_prefixes[spec.kind] + base_name,
            is_placeholder=True,
        )
        if base_name in custom_meta:
            # the keys in custom_meta are node names from `mod`,
            # which is the base_name here.
            # we need the re-mapped name for lookup later
            custom_meta[name_map[spec.arg.name]] = custom_meta[base_name]
            del custom_meta[base_name]

    # handle naming collisions with call_function/get_attr inputs.
    # here, we want to prioritize user input names over call_function names
    # e.g. not have forward(self, mul): lead to a placeholder node called mul_13,
    # so we increment the suffix of call_function nodes as needed
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue
        _rename_without_collisions(
            name_map, find_available, used_names, node.name, node.name
        )

    # assign new node names
    _assign_new_node_names(gm, name_map, custom_meta)

    # propagate names to higher order op subgraphs
    _name_hoo_subgraph_placeholders(gm)

    # re-generate graph module code
    gm.recompile()

    # modify graph signature (input specs, output specs, user input mutations)
    for spec in export_graph_signature.input_specs:
        assert spec.arg.name in name_map
        spec.arg.name = name_map[spec.arg.name]
        if (  # handle targets for custom objects
            spec.kind == InputKind.CUSTOM_OBJ and spec.target in name_map
        ):
            # pyrefly: ignore [bad-index, index-error]
            spec.target = name_map[spec.target][4:]  # strip obj_ prefix

    for spec in export_graph_signature.output_specs:
        if spec.arg.name in name_map:
            spec.arg.name = name_map[spec.arg.name]
        if spec.kind == OutputKind.USER_INPUT_MUTATION and spec.target in name_map:
            # pyrefly: ignore [bad-index, index-error]
            spec.target = name_map[spec.target]

    # rename keys in constants dict for custom objects
    for name in list(constants.keys()):
        constant = constants[name]
        if name in name_map and not isinstance(
            constant, torch.Tensor
        ):  # rename custom objects with generic names
            new_name = name_map[name]
            if (
                new_name != name
                and re.match(r"arg(\d+)_1", name)
                and new_name != placeholder_prefixes[InputKind.CUSTOM_OBJ] + name
            ):
                constants[new_name] = constant
                del constants[name]


def remove_proxy_from_state_dict(state_dict: dict, in_place: bool) -> dict:
    """
    If `in_place` is false, return a new copy of `state_dict` with "proxy" removed from `v.__dict__`.
    `v` is the values in the dictionary.
    If `in_place` is true, modify `state_dict` in place.
    """
    if in_place:
        for k, v in state_dict.items():
            if hasattr(v, "proxy"):
                delattr(state_dict[k], "proxy")
        return state_dict
    else:
        new_state_dict = {}
        for k, v in state_dict.items():
            if hasattr(v, "proxy"):
                new_state_dict[k] = v.detach().clone()
            else:
                new_state_dict[k] = v
        return new_state_dict


def _detect_fake_mode_from_gm(
    gm: torch.fx.GraphModule,
) -> Optional[torch._subclasses.fake_tensor.FakeTensorMode]:
    """
    For a given graph module, we look at the "val" of placeholder nodes to find the fake inputs.
    Additionally, if gm doesn't have placeholders, we further look at the "example_value" or "val" of other nodes.
    If no fake mode is found, we return None for fake_mode.
    """

    fake_inps: list[torch.Tensor] = []
    fake_vals: list[torch.Tensor] = []
    for node in gm.graph.nodes:
        if node.op == "placeholder" and "val" in node.meta:
            fake_val = node.meta["val"]
            if fake_val is not None and isinstance(fake_val, torch.Tensor):
                fake_inps.append(fake_val)
        elif len(fake_inps) == 0 and (
            "example_value" in node.meta or "val" in node.meta
        ):
            fake_val = None
            if "example_value" in node.meta:
                fake_val = node.meta["example_value"]
            elif "val" in node.meta:
                fake_val = node.meta["val"]
            if fake_val is not None and isinstance(fake_val, torch.Tensor):
                fake_vals.append(fake_val)

    return detect_fake_mode(fake_inps + fake_vals)


@contextmanager
def _disable_load_state_dict_hooks(mod: torch.nn.Module):
    state_dict_hooks: dict[int, Callable] = dict(mod._state_dict_hooks)
    state_dict_pre_hooks: dict[int, Callable] = dict(mod._state_dict_pre_hooks)
    mod._state_dict_hooks.clear()
    mod._state_dict_pre_hooks.clear()
    try:
        yield
    finally:
        mod._state_dict_hooks = state_dict_hooks
        mod._state_dict_pre_hooks = state_dict_pre_hooks


def _is_cia_op(op: "OperatorBase") -> bool:
    return (
        torch._C._dispatch_has_kernel_for_dispatch_key(
            op.name(), torch._C.DispatchKey.CompositeImplicitAutograd
        )
        or torch._C.DispatchKey.CompositeImplicitAutograd in op.py_kernels
    )


def _is_preservable_cia_op(op: "OperatorBase") -> bool:
    return _check_valid_to_preserve(op) and _is_cia_op(op)


def _is_aten_op(op: "OperatorBase") -> bool:
    return op.name().split("::")[0] == "aten"


def _is_custom_op(op: "OperatorBase") -> bool:
    return not _is_aten_op(op)


# We can't cache this because custom op registry API in python can still
# add entries to the C++ dispatcher.
def _materialize_cpp_cia_ops() -> None:
    """
    Utility function to query C++ dispatcher to get the all
    possible CIA ops and populate them into torch.ops namespace
    """
    cia_ops = torch._C._dispatch_get_registrations_for_dispatch_key(
        "CompositeImplicitAutograd"
    )

    # Materialize all CIA ops
    for op in cia_ops:
        namespace, op_name = tuple(op.split("::"))
        split_list = op_name.split(".")
        # Sometime overload could be missing
        assert len(split_list) == 1 or len(split_list) == 2
        op_name = split_list[0]
        op_overload_name = "default"
        if len(split_list) == 2:
            op_overload_name = split_list[1]

        _ = getattr(getattr(getattr(torch.ops, namespace), op_name), op_overload_name)


def _special_op_to_preserve_cia(*args, **kwargs):
    """
    This is an special marker that tells our infra that we shouldn't decompose this op.
    """
    return NotImplemented


# Our strategy for deciding if we can preserve a op is following:
# 1. The op should be known statically that it is functional
# 2. If it is maybe aliasing, we decompose because we must know if an op
#    is mutating or aliasing.
def _check_valid_to_preserve(op_overload: "OperatorBase"):
    from torch._decomp import _should_decompose_because_unsafe_op

    if _should_decompose_because_unsafe_op(op_overload):
        return False
    if op_overload in FunctionalTensor.metadata_fns:
        return False

    if not hasattr(op_overload, "_schema"):
        return False

    alias_info = len(
        [i for i in op_overload._schema.arguments if i.alias_info is not None]
    )

    is_mutating_or_aliasing = alias_info != 0 or op_overload._schema.is_mutable

    if is_mutating_or_aliasing:
        return False

    if not torch._C._dispatch_has_kernel(op_overload.name()):
        return False

    return True


@functools.lru_cache(maxsize=1)
def _collect_all_valid_cia_ops_for_aten_namespace() -> set["OperatorBase"]:
    return _collect_all_valid_cia_ops_for_namespace(torch.ops.aten)


def _collect_all_valid_cia_ops_for_namespace(
    op_namespace: torch._ops._OpNamespace,
) -> set["OperatorBase"]:
    # Step 1: Materialize all ops from C++ dispatcher
    _materialize_cpp_cia_ops()

    # Step 2: Query all ops from python dispatcher
    cia_ops = set()
    for op in op_namespace:
        op_packet = getattr(op_namespace, op)
        for overload in op_packet.overloads():
            op_overload = getattr(op_packet, overload)
            if _is_preservable_cia_op(op_overload):
                cia_ops.add(op_overload)
    return cia_ops


def _collect_all_valid_cia_ops() -> set["OperatorBase"]:
    """
    This is an util function that gets the all CIA functional ops.

    The algorithm is in 2 steps:
      1. We first query C++ dispatcher to get the list of CIA ops
         and then we call getattr on torch.ops.aten to lazily populate
         them.

      2. Sometimes, handful of ops have CIA registered in python dispatcher
         but not on the C++ side, these can't be caught at the first step.
         So we walk again to get the final list.

    Note that the output of this function should never be modified
    """
    cia_ops = set()
    for op_namespace_name in torch.ops._dir:
        # The reason we split here is because aten ops are safe to cache.
        if op_namespace_name != "aten":
            assert hasattr(torch.ops, op_namespace_name)
            op_namespace = getattr(torch.ops, op_namespace_name)
            if isinstance(op_namespace, torch._ops._OpNamespace):
                cia_ops |= _collect_all_valid_cia_ops_for_namespace(op_namespace)
        else:
            cia_ops |= _collect_all_valid_cia_ops_for_aten_namespace()
    return cia_ops


def _get_decomp_for_cia(op: "OperatorBase"):
    # [NOTE] Separating out func.decompose
    # Ideally we should be able to just register func.decompose but
    # we can't as this decomp is gonna be registered to the py_impl.
    # As a result it will infinitely recurse. So we first check if the op
    # has py_impl entry for CIA and if it is we use that first. If not,
    # we register C++ query to py_impl.
    dk = torch._C.DispatchKey.CompositeImplicitAutograd
    if dk in op.py_kernels and not isinstance(op.py_kernels[dk], torch._C.DispatchKey):
        return op.py_kernels[dk]

    def _special_op_to_decompose_cia(*args, **kwargs):
        kernel = kwargs["kernel"]
        del kwargs["kernel"]
        # Can't call kernel.decompose due to infinite recursion as
        # we register this kernel to py_impl directly
        dk = torch._C.DispatchKey.CompositeImplicitAutograd
        if torch._C._dispatch_has_kernel_for_dispatch_key(
            kernel.name(), torch._C.DispatchKey.CompositeImplicitAutograd
        ):
            return kernel._op_dk(dk, *args, **kwargs)
        else:
            raise AssertionError(
                f"Expected {kernel} to have CompositeImplicitAutograd kernel"
            )

    return functools.partial(_special_op_to_decompose_cia, kernel=op)


@contextmanager
def _compiling_state_context():
    old_compiling_flag = torch.compiler._is_compiling_flag
    old_exporting_flag = torch.compiler._is_exporting_flag
    try:
        torch.compiler._is_compiling_flag = True
        torch.compiler._is_exporting_flag = True
        yield
    finally:
        torch.compiler._is_compiling_flag = old_compiling_flag
        torch.compiler._is_exporting_flag = old_exporting_flag


def _fakify_params_buffers(
    fake_mode: FakeTensorMode,
    mod: torch.nn.Module,
) -> dict[str, Union[torch.Tensor, torch.nn.Parameter]]:
    params_buffers = {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }

    faked_params_buffers = {}
    memo: dict[int, FakeTensor] = {}
    for key, value in params_buffers.items():
        if id(value) in memo:
            fake_tensor = memo[id(value)]
        else:
            fake_tensor = fake_mode.from_tensor(value, static_shapes=True)
            memo[id(value)] = fake_tensor
        faked_params_buffers[key] = fake_tensor
    return faked_params_buffers  # type: ignore[return-value]


def register_module_as_pytree_input_node(cls: type[torch.nn.Module]) -> None:
    """
    Registers a module as a valid input type for :func:`torch.export.export`.

    Args:
        mod: the module instance
        serialized_type_name: The serialized name for the module. This is
        required if you want to serialize the pytree TreeSpec containing this
        module.

    Example::

        import torch


        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return self.linear(x)


        torch._export.utils.register_module_as_pytree_node(InputDataClass)


        class Mod(torch.nn.Module):
            def forward(self, x, m):
                return m(x) + x


        ep = torch.export.export(Mod(), (torch.randn(3), Module()))
        print(ep)

    """
    assert issubclass(cls, torch.nn.Module)

    import weakref

    class PrototypeModule(weakref.ref):
        def __init__(self, m, *args, **kwargs):
            super().__init__(m, *args, **kwargs)  # type: ignore[call-arg]
            assert isinstance(m, torch.nn.Module)
            assert not hasattr(self, "_proto_cls")
            self._proto_cls = cls

        def __eq__(self, other):
            return self._proto_cls == other._proto_cls

        def __deepcopy__(self, memo):
            return PrototypeModule(self())

    def default_flatten_fn(obj: Any) -> tuple[list[Any], Context]:
        named_parameters = dict(obj.named_parameters())
        named_buffers = dict(obj.named_buffers())
        params_buffers = {**named_parameters, **named_buffers}
        return list(params_buffers.values()), [
            list(params_buffers.keys()),
            PrototypeModule(obj),
        ]

    def default_unflatten_fn(values: Iterable[Any], context: Context) -> Any:
        flat_names, ref = context
        if ref is None or ref() is None:
            raise RuntimeError("Module has been garbage collected")
        obj = ref()
        assert flatten_fn is not None
        flattened, _ = flatten_fn(obj)

        # NOTE: This helper function will replicate an nn.Module in the exactly same
        #       structure to be used together with _reparameterize_module. This will
        #       create a clone of the module with the new parameters and buffers without
        #       affecting the original module.
        def copy_module(mod: torch.nn.Module):
            ret = copy.copy(mod)
            ret.__dict__ = {copy.copy(k): copy.copy(v) for k, v in mod.__dict__.items()}
            for name, child in ret.named_children():
                setattr(ret, name, copy_module(child))
            return ret

        if any(v is not o for v, o in zip(values, flattened)):
            with torch.nn.utils.stateless._reparametrize_module(
                obj, dict(zip(flat_names, values)), tie_weights=True, strict=True
            ):
                ret = copy_module(obj)
        else:
            ret = obj
        return ret

    def default_flatten_fn_with_keys(obj: Any) -> tuple[list[Any], Context]:
        flattened, [flat_names, *args] = flatten_fn(obj)  # type: ignore[misc]
        return [(MappingKey(k), v) for k, v in zip(flat_names, flattened)], [
            flat_names,
            *args,
        ]

    flatten_fn = default_flatten_fn
    unflatten_fn = default_unflatten_fn

    serialized_type_name = cls.__module__ + "." + cls.__qualname__

    def to_dumpable_context(context):
        keys, *_ = context
        return json.dumps([keys, *([None] * len(_))])

    def from_dumpable_context(dumpable):
        s = json.loads(dumpable)
        s[1] = PrototypeModule(torch.nn.Module())
        return s

    _register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        flatten_with_keys_fn=default_flatten_fn_with_keys,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
    )

    def default_flatten_fn_spec(obj, spec) -> list[Any]:
        flats, context = flatten_fn(obj)
        assert context == spec.context
        return flats

    register_pytree_flatten_spec(
        cls,
        default_flatten_fn_spec,
    )


def deregister_module_as_pytree_input_node(cls: type[torch.nn.Module]) -> None:
    _deregister_pytree_node(cls)
    _deregister_pytree_flatten_spec(cls)


def _sync_state(src, dst):
    assert isinstance(
        src,
        torch.nn.Module,
    ), f"Expected {src} to be a nn.Module"
    assert isinstance(
        dst,
        torch.nn.Module,
    ), f"Expected {dst} to be a nn.Module"
    # Share state (params, buffers) between modules.
    # This ensures that state mutations are visible across them.
    # Since tensor constants are not mutable, copying (without sharing) is OK.
    # Also, primitive constants are specialized, so copying (without sharing) is OK.
    dst._parameters = src._parameters
    dst._buffers = src._buffers


def sync_state(*wrapped_method_modules):
    """
    Sync state between exported modules corresponding to wrapped methods.
    This might be necessary after serializing/deserializing due to copying.
    """
    if wrapped_method_modules:
        m, *other_ms = wrapped_method_modules
        for other_m in other_ms:
            _sync_state(m, other_m)


class _WrappedMethod(torch.nn.Module):
    def __init__(self, method):
        super().__init__()
        # share state of method's self module
        _sync_state(method.__self__, self)
        # redirect forward to method
        self.forward = method


def wrap_method(method):
    """
    Wrap a method as a module so that it can be exported.
    The wrapped module's forward points to the method, and
    the method's original module state is shared.
    """
    assert ismethod(
        method,
    ), f"Expected {method} to be a method"
    return _WrappedMethod(method)
