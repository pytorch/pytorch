import dataclasses

from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from torch._export import ExportedProgram

from torch.utils._pytree import (
    _register_pytree_node,
    Context,
    DumpableContext,
    FlattenFunc,
    FromDumpableContextFn,
    ToDumpableContextFn,
    UnflattenFunc,
)


SERIALIZED_DATACLASS_TO_PYTHON_DATACLASS: Dict[str, Type[Any]] = {}


def register_dataclass_as_pytree_node(
    typ: Any,
    flatten_fn: Optional[FlattenFunc] = None,
    unflatten_fn: Optional[UnflattenFunc] = None,
    *,
    to_dumpable_context: Optional[ToDumpableContextFn] = None,
    from_dumpable_context: Optional[FromDumpableContextFn] = None,
    return_none_fields: bool = False,
) -> None:
    assert dataclasses.is_dataclass(
        typ
    ), f"Only dataclasses can be registered with this function: {typ}"

    serialized_type = f"{typ.__module__}.{typ.__name__}"
    SERIALIZED_DATACLASS_TO_PYTHON_DATACLASS[serialized_type] = typ

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
        return flattened, (typ, flat_names, none_names)

    def default_unflatten_fn(values: List[Any], context: Context) -> Any:
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
            f"Both to_dumpable_context and from_dumpable_context for {typ} must "
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
        typ,
        flatten_fn,
        unflatten_fn,
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
