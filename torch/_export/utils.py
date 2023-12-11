import dataclasses

from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import torch

from torch._export import ExportedProgram

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
        self.graph,
        range_constraints=self.range_constraints,
        equality_constraints=self.equality_constraints,
    )(*flat_args)


def _check_input_constraints_for_graph(
    graph: torch.fx.Graph, range_constraints, equality_constraints
):
    from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
        _AddRuntimeAssertionsForConstraintsPass,
    )

    def inner(*args):
        # TODO(zhxchen17) Don't generate a runtime graph on the fly.
        _assertion_graph = torch.fx.GraphModule({}, torch.fx.Graph())
        for p in graph.nodes:
            if p.op != "placeholder":
                continue
            new_p = _assertion_graph.graph.placeholder(p.name)
            new_p.meta = p.meta
        _assertion_graph.graph.output(())
        _assertion_graph_res = _AddRuntimeAssertionsForConstraintsPass(
            range_constraints,
            equality_constraints,
        )(_assertion_graph)
        assert _assertion_graph_res is not None
        _assertion_graph = _assertion_graph_res.graph_module
        _assertion_graph(*args)

    return inner


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
