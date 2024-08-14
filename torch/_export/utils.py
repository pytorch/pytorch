# mypy: allow-untyped-defs
import ast
import dataclasses
import inspect
import math
import operator
import re
from inspect import Parameter
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TYPE_CHECKING

import torch
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensor


if TYPE_CHECKING:
    from torch._export.passes.lift_constants_pass import ConstantAttrMap
    from torch.export import ExportedProgram
    from torch.export.graph_signature import ExportGraphSignature

from torch.export.graph_signature import InputKind, OutputKind
from torch.utils._pytree import (
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


def _collect_and_set_constant_attrs(
    graph_signature, constants, mod
) -> "ConstantAttrMap":
    # the exported module will store constants & non-persistent buffers such that
    # retracing treats them as persistent buffers, so we inform the constants lifting pass
    # and overwrite the new graph signature using the previous program.
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


def _collect_param_buffer_metadata(mod: torch.fx.GraphModule) -> Dict[str, Any]:
    """
    Param/buffer metadata needs to be saved before lowering to aten IR
    because aten IR lifts them, as a result, automatic preservation doesn't work
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
                        if entry in meta:
                            params_buffers_to_node_meta[arg.target][entry] = meta[entry]

    return params_buffers_to_node_meta


def _populate_param_buffer_metadata_to_new_gm(
    params_buffers_to_node_meta: Dict[str, Any],
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


def _check_input_constraints_for_graph(
    input_placeholders: List[torch.fx.Node], flat_args_with_path, range_constraints
):
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
            assert isinstance(kwarg_key, MappingKey)
            name = str(kwarg_key)[1:-1]  # get rid of the enclosed []
            return f"{name}{keystr(key_path[2:])}"

    import sympy

    from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
        _convert_range_to_int,
    )
    from torch.utils._sympy.solve import try_solve

    if len(flat_args_with_path) != len(input_placeholders):
        raise RuntimeError(
            "Unexpected number of inputs "
            f"(expected {len(input_placeholders)}, got {len(flat_args_with_path)})"
        )
    # NOTE: export already guarantees that the same symbol is used in metadata
    # for all InputDims related by equality constraints, so we can just unify
    # symbols with given input dimension values to check equality constraints.
    unification_map: Dict[sympy.Symbol, Any] = {}
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
                # TODO(avik): Assert the following property in the IR verifier:
                # node_dim is either an int or a SymInt containing an int or a unary sympy.Expr
                if (
                    isinstance(node_dim, torch.SymInt)
                    and len(node_dim.node.expr.free_symbols) == 1
                ):
                    symbol = next(iter(node_dim.node.expr.free_symbols))
                    if symbol in unification_map:
                        existing_dim = node_dim.node.expr.subs(unification_map)
                        if arg_dim != existing_dim:
                            raise RuntimeError(
                                f"Expected input at {get_keystr(key_path)}.shape[{j}] to be equal to "
                                f"{existing_dim}, but got {arg_dim}",
                            )
                    else:
                        if (
                            isinstance(arg_dim, torch.SymInt)
                            and not arg_dim.node.expr.is_number
                        ):
                            # This can happen when, say, arg is a fake tensor.
                            # We do not run checks on symbolic shapes of fake inputs as
                            # such checks can affect the shape env.
                            pass
                        else:
                            if isinstance(node_dim.node.expr, sympy.Symbol):
                                # Short cut for try_solve below. Also useful in cases where
                                # sympy.Eq(node_dim.node.expr, arg_dim) would evaluate to False
                                # purely because symbol is constrained to be size-like,
                                # e.g., when node_dim.node.expr = symbol and arg_dim = 0.
                                unification_map[symbol] = int(arg_dim)
                            else:
                                solution = try_solve(
                                    sympy.Eq(node_dim.node.expr, arg_dim), symbol
                                )
                                if solution is None:
                                    raise RuntimeError(  # noqa: B904
                                        f"Expected input {node.name}.shape[{j}] = {arg_dim} to be "
                                        f"of the form {node_dim.node.expr}, where {symbol} is an integer"
                                    )
                                else:
                                    unification_map[symbol] = int(solution[1])

                    if node_dim.node.expr in range_constraints:
                        min_val, max_val = _convert_range_to_int(
                            range_constraints[node_dim.node.expr]
                        )
                        # NOTE: we allow dimensions to be 0/1 at runtime
                        if min_val > 2:
                            if arg_dim < min_val:
                                raise RuntimeError(
                                    f"Expected input at {get_keystr(key_path)}.shape[{j}] to be >= "
                                    f"{min_val}, but got {arg_dim}",
                                )
                        if max_val < math.inf:
                            if arg_dim > max_val:
                                raise RuntimeError(
                                    f"Expected input at {get_keystr(key_path)}.shape[{j}] to be <= "
                                    f"{max_val}, but got {arg_dim}",
                                )
                else:
                    if arg_dim != node_dim:
                        if isinstance(node_dim, torch.SymInt):
                            # this means we deferred a guard from export analysis to runtime, let this pass
                            # we'll add a runtime assert checking equality to this replacement expression
                            continue
                        raise RuntimeError(
                            f"Expected input at {get_keystr(key_path)}.shape[{j}] to be equal to "
                            f"{node_dim}, but got {arg_dim}",
                        )
        elif isinstance(node_val, (int, float, str)):
            if type(arg) != type(node_val) or arg != node_val:
                raise RuntimeError(
                    f"Expected input at {get_keystr(key_path)} to be equal to {node_val}, but got {arg}",
                )


def register_dataclass_as_pytree_node(
    cls: Type[Any],
    flatten_fn: Optional[FlattenFunc] = None,
    unflatten_fn: Optional[UnflattenFunc] = None,
    *,
    serialized_type_name: Optional[str] = None,
    to_dumpable_context: Optional[ToDumpableContextFn] = None,
    from_dumpable_context: Optional[FromDumpableContextFn] = None,
    return_none_fields: bool = False,
) -> None:
    assert dataclasses.is_dataclass(
        cls
    ), f"Only dataclasses can be registered with this function: {cls}"

    def default_flatten_fn(obj: Any) -> Tuple[List[Any], Context]:
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

    def default_unflatten_fn(values: Iterable[Any], context: Context) -> Any:
        flat_names, none_names = context
        return cls(**dict(zip(flat_names, values)), **dict.fromkeys(none_names))

    def default_flatten_fn_with_keys(obj: Any) -> Tuple[List[Any], Context]:
        flattened, (flat_names, none_names) = flatten_fn(obj)  # type: ignore[misc]
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


def sequential_split(gm: torch.fx.GraphModule, node_call_back) -> torch.fx.GraphModule:
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


def nodes_filter(nodes: List[torch.fx.Node], node_call_back) -> List[torch.fx.Node]:
    """Returns the nodes that match the node_call_back as a list."""
    return [node for node in nodes if node_call_back(node)]


def nodes_first(
    nodes: List[torch.fx.Node], node_call_back=None
) -> Optional[torch.fx.Node]:
    """
    Returns the first node that matches the node_call_back. If no node matches, returns None.
    When node_call_back is None, returns the first node in the node list.
    """
    ret = nodes_filter(nodes, node_call_back if node_call_back else lambda node: True)
    if len(ret) > 0:
        return ret[0]
    return None


def nodes_count(nodes: List[torch.fx.Node], node_call_back) -> int:
    """Returns the number of nodes that match the node_call_back."""
    return len(nodes_filter(nodes, node_call_back))


def nodes_map(nodes: List[torch.fx.Node], node_call_back) -> List[torch.fx.Node]:
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


def node_inline_(call_mod_node: torch.fx.Node) -> None:
    """
    Inline the submodule of the given node into the parent module.
    Note: we only support the case where submodule takes tensors inputs.
    """
    assert call_mod_node.op == "call_module"
    gm = call_mod_node.graph.owning_module

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
                # Clear the users of the output node and set
                # the users to be the users of original call_module node.
                for node in new_output:
                    node.users.clear()

                # Inline the get_item calls for the output node.
                get_item_users = nodes_filter(
                    list(call_mod_node.users.keys()),
                    lambda node: node.op == "call_function"
                    and node.target == operator.getitem,
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


def _get_torch_jit_trace_forward_signature(mod: torch.nn.Module):
    """
    Get source code and parse argument names using AST. The function returns
    a signature of the forward() function.

    # TODO: Directly provide inspect.signature compatible TS-d module.
    """
    ast_mod = ast.parse(mod.code)
    ast_func_def: ast.FunctionDef = ast_mod.body[0]  # type: ignore[assignment]

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

    return sig.bind(*fake_args, **fake_kwargs).arguments


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


def placeholder_naming_pass(
    gm: torch.fx.GraphModule,
    export_graph_signature: "ExportGraphSignature",
    mod: torch.nn.Module,
    fake_args,
    fake_kwargs,
    fake_params_buffers,
    constants: Dict[str, Any],
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

    def _strip_name(x):
        if x.startswith("L__self___"):
            x = x[len("L__self___") :]
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

    name_map: Dict[str, str] = {}

    # map user input names with mod.forward() signature
    combined_args = _bind_signature_to_inputs(mod, fake_args, fake_kwargs)

    flat_args_with_path, _ = tree_flatten_with_path(combined_args)
    user_input_names = [
        spec.arg.name
        for spec in export_graph_signature.input_specs
        if spec.kind == InputKind.USER_INPUT
    ]

    # use pytree path to name nested user inputs
    for (arg_path, arg), user_input_name in zip(flat_args_with_path, user_input_names):
        if user_input_name:
            _rename_without_collisions(
                name_map,
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
            spec.arg.name,
            placeholder_prefixes[spec.kind] + base_name,
            is_placeholder=True,
        )

    # handle naming collisions with call_function/get_attr inputs.
    # here, we want to prioritize user input names over call_function names
    # e.g. not have forward(self, mul): lead to a placeholder node called mul_13,
    # so we increment the suffix of call_function nodes as needed
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue
        _rename_without_collisions(name_map, node.name, node.name)

    # assign new node names
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            assert node.name in name_map
            node.name = node.target = name_map[node.name]
        elif node.name in name_map:
            node.name = name_map[node.name]

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
            spec.target = name_map[spec.target][4:]  # strip obj_ prefix

    for spec in export_graph_signature.output_specs:
        if spec.arg.name in name_map:
            spec.arg.name = name_map[spec.arg.name]
        if spec.kind == OutputKind.USER_INPUT_MUTATION and spec.target in name_map:
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


def remove_proxy_from_state_dict(state_dict: Dict, in_place: bool) -> Dict:
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
                new_state_dict[k] = v.clone().detach()
            else:
                new_state_dict[k] = v
        return new_state_dict


def _detect_fake_mode_from_gm(
    gm: torch.fx.GraphModule,
) -> torch._subclasses.fake_tensor.FakeTensorMode:
    """
    For a given graph module, we look at the "val" of placeholder nodes to find the fake inputs.
    Additionally, if gm doesn't have placeholders, we further look at the "example_value" or "val" of other nodes.
    If no fake mode is found, we return None for fake_mode.
    """

    fake_inps: List[torch.Tensor] = []
    fake_vals: List[torch.Tensor] = []
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
