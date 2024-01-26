import dataclasses
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import torch

from torch._export import ExportedProgram
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._pytree import (
    _register_pytree_node,
    Context,
    DumpableContext,
    FlattenFunc,
    FromDumpableContextFn,
    ToDumpableContextFn,
    tree_flatten,
    UnflattenFunc,
)


SERIALIZED_DATACLASS_TO_PYTHON_DATACLASS: Dict[str, Type[Any]] = {}


@torch._dynamo.disable
def _check_input_constraints_pre_hook(self, *args, **kwargs):
    flat_args, _ = tree_flatten(args)
    return _check_input_constraints_for_graph(
        [node for node in self.graph.nodes if node.op == "placeholder"],
        flat_args,
        self.range_constraints,
    )


def _check_input_constraints_for_graph(
    input_placeholders: List[torch.fx.Node], args, range_constraints
):
    def check(cond, msg):
        if not cond:
            # TODO(avik): maybe add more context, e.g., graph signature
            raise RuntimeError(msg)

    import sympy

    from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
        _convert_range_to_int,
    )

    check(
        len(args) == len(input_placeholders),
        "Unexpected number of inputs "
        f"(expected {len(input_placeholders)}, got {len(args)})",
    )
    # NOTE: export already guarantees that the same symbol is used in metadata
    # for all InputDims related by equality constraints, so we can just unify
    # symbols with given input dimension values to check equality constraints.
    unification_map: "Dict[sympy.Symbol, Any]" = {}
    for arg, node in zip(args, input_placeholders):
        node_val = node.meta.get("val")
        if isinstance(node_val, FakeTensor):
            check(
                isinstance(arg, torch.Tensor),
                f"Expected input {node.name} to be a tensor, but got {type(arg)}",
            )
            check(
                len(node_val.shape) == len(arg.shape),
                f"Unexpected number of dimensions in input {node.name}.shape "
                f"(expected {node_val.shape}, got {arg.shape})",
            )
            for j, (arg_dim, node_dim) in enumerate(zip(arg.shape, node_val.shape)):
                if isinstance(node_dim, torch.SymInt):
                    if node_dim.node.expr in unification_map:
                        existing_dim = unification_map[node_dim.node.expr]
                        check(
                            arg_dim == existing_dim,
                            f"Expected input {node.name}.shape[{j}] to be equal to "
                            f"{existing_dim}, but got {arg_dim}",
                        )
                    else:
                        unification_map[node_dim.node.expr] = arg_dim

                    if node_dim.node.expr in range_constraints:
                        min_val, max_val = _convert_range_to_int(
                            range_constraints[node_dim.node.expr]
                        )
                        # NOTE: we allow dimensions to be 0/1 at runtime
                        if min_val > 2:
                            check(
                                arg_dim >= min_val,
                                f"Expected input {node.name}.shape[{j}] to be >= "
                                f"{min_val}, but got {arg_dim}",
                            )
                        if max_val < math.inf:
                            check(
                                arg_dim <= max_val,
                                f"Expected input {node.name}.shape[{j}] to be <= "
                                f"{max_val}, but got {arg_dim}",
                            )
                else:
                    check(
                        arg_dim == node_dim,
                        f"Expected input {node.name}.shape[{j}] to be equal to "
                        f"{node_dim}, but got {arg_dim}",
                    )
        elif isinstance(node_val, (int, float, str)):
            check(
                type(arg) == type(node_val) and arg == node_val,
                f"Expected input {node.name} to be equal to {node_val}, but got {arg}",
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

    serialized_type = f"{cls.__module__}.{cls.__qualname__}"
    SERIALIZED_DATACLASS_TO_PYTHON_DATACLASS[serialized_type] = cls

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
        return flattened, (cls, flat_names, none_names)

    def default_unflatten_fn(values: Iterable[Any], context: Context) -> Any:
        typ, flat_names, none_names = context
        return typ(**dict(zip(flat_names, values)), **{k: None for k in none_names})

    def default_to_dumpable_context(context: Context) -> DumpableContext:
        return (serialized_type, context[1], context[2])

    def default_from_dumpable_context(dumpable_context: DumpableContext) -> Context:
        return (
            SERIALIZED_DATACLASS_TO_PYTHON_DATACLASS[dumpable_context[0]],
            dumpable_context[1],
            dumpable_context[2],
        )

    flatten_fn = flatten_fn if flatten_fn is not None else default_flatten_fn
    unflatten_fn = unflatten_fn if unflatten_fn is not None else default_unflatten_fn

    if (to_dumpable_context is None) ^ (from_dumpable_context is None):
        raise ValueError(
            f"Both to_dumpable_context and from_dumpable_context for {cls} must "
            "be None or registered."
        )

    to_dumpable_context = (
        to_dumpable_context
        if to_dumpable_context is not None
        else default_to_dumpable_context
    )
    from_dumpable_context = (
        from_dumpable_context
        if from_dumpable_context is not None
        else default_from_dumpable_context
    )

    _register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
    )


def is_param(program: ExportedProgram, node: torch.fx.Node) -> bool:
    """
    Checks if the given node is a parameter within the exported program
    """

    return node.name in program.graph_signature.inputs_to_parameters


def get_param(
    program: ExportedProgram,
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


def is_buffer(program: ExportedProgram, node: torch.fx.Node) -> bool:
    """
    Checks if the given node is a buffer within the exported program
    """

    return node.name in program.graph_signature.inputs_to_buffers


def get_buffer(
    program: ExportedProgram,
    node: torch.fx.Node,
) -> Optional[torch.Tensor]:
    """
    Returns the buffer associated with the given node in the exported program.
    Returns None if the node is not a buffer within the exported program
    """

    if is_buffer(program, node):
        buffer_name = program.graph_signature.inputs_to_buffers[node.name]
        return program.state_dict[buffer_name]

    return None
