# mypy: allow-untyped-defs
"""
Effect ordering pass for inductor.

This pass adds ordering dependencies to FX graphs using the control_deps HOP
for precise control over scheduling constraints. When you need exact ordering between
operations (e.g., collective_start -> mm -> wait), this pass wraps operations
with control_deps to make dependencies explicit.
"""

from typing import Any

import torch.fx as fx
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch.utils._ordered_set import OrderedSet


class ControlDeps(HigherOrderOperator):
    """
    Higher-order operator that enforces ordering by making dependencies explicit.

    Schema: control_deps(additional_deps, target, *args, **kwargs) -> result
    where:
    - additional_deps: tuple of tensors that must be computed before this op
    - subgraph: GraphModule containing the exact operation to execute
    - args/kwargs: arguments for the target function

    This ensures all tensors in additional_deps are computed before the target
    executes, creating explicit scheduling dependencies.
    """

    def __init__(self) -> None:
        super().__init__("control_deps")

    def __call__(self, additional_deps, subgraph, *args, **kwargs):
        """Call the operator with dependencies and subgraph.

        Args:
            additional_deps: Tuple of tensors that must be computed first
            subgraph: GraphModule containing the exact operation to execute
            *args: Arguments to pass to the subgraph
        """
        if not isinstance(additional_deps, (tuple, list)):
            raise TypeError(
                f"additional_deps must be tuple/list, got {type(additional_deps).__name__}"
            )
        if not (isinstance(subgraph, fx.GraphModule) or callable(subgraph)):
            raise TypeError(
                f"subgraph must be GraphModule or callable, got {type(subgraph).__name__}"
            )
        return super().__call__(additional_deps, subgraph, *args, **kwargs)


control_deps = ControlDeps()


# Register fake implementation for tracing
@register_fake(control_deps)
def _(additional_deps, subgraph, *args, **kwargs):
    """Fake tensor implementation - execute the subgraph."""
    return subgraph(*args, **kwargs)


def get_subgraph_name(gm: fx.GraphModule, name):
    name = f"subgraph_{name}"

    if not hasattr(gm, name):
        return name

    i = 0
    while hasattr(gm, f"{name}_{i}"):
        i += 1

    return f"{name}_{i}"


def preserve_node_ordering(
    graph: fx.Graph,
    additional_deps_map: dict[fx.Node, OrderedSet[fx.Node]],
    verbose: bool = False,
) -> None:
    """
    Preserve node ordering using control_deps HOP with subgraph.

    This function wraps operations with control_deps that:
    1. Makes additional dependencies explicit (first argument)
    2. Creates a subgraph internally to preserve the exact original operation
    3. Preserves the original node names

    Args:
        graph: The FX graph to modify
        additional_deps_map: Mapping from dependent nodes to their dependencies
        verbose: If True, print debug information
    """
    if not additional_deps_map:
        return

    # Track replacements so we can update dependencies
    replacements: dict[fx.Node, fx.Node] = {}

    # Process each node that needs additional dependencies
    for dependent_node, dep_nodes in additional_deps_map.items():
        assert dependent_node.op == "call_function", dependent_node.op

        original_name = dependent_node.name
        original_args = dependent_node.args
        original_kwargs = dependent_node.kwargs
        original_meta = dependent_node.meta.copy()

        updated_dep_nodes = [replacements.get(dep, dep) for dep in dep_nodes]

        # Create a subgraph that preserves the exact original operation
        subgraph_module = _create_subgraph_for_node(graph, dependent_node)

        owning_mod = graph.owning_module
        assert owning_mod is not None
        subgraph_attr_name = get_subgraph_name(owning_mod, original_name)
        setattr(graph.owning_module, subgraph_attr_name, subgraph_module)

        # Create control_deps call with:
        # 1. Additional dependencies as first arg (explicit)
        # 2. Subgraph via get_attr (like b2b gemm pass)
        # 3. Original arguments (only fx.Node args and kwargs are passed)
        with graph.inserting_before(dependent_node):
            # Create get_attr node for the subgraph
            get_subgraph = graph.get_attr(subgraph_attr_name)

            # add additional args
            node_args = [a for a in original_args if isinstance(a, fx.Node)]
            for value in original_kwargs.values():
                if isinstance(value, fx.Node):
                    node_args.append(value)

            # Create with temporary name first
            ordered_node = graph.call_function(
                control_deps,
                args=(
                    tuple(updated_dep_nodes),  # additional_deps
                    get_subgraph,  # subgraph via get_attr (like b2b gemm)
                    *node_args,  # original node arguments (from both args and kwargs)
                ),
                kwargs={},
                name=f"__temp_{original_name}",  # Temporary name to avoid conflict
            )

        # Copy metadata from original node
        ordered_node.meta = original_meta
        # this will be constrained on the target node in subgraph if it exists
        ordered_node.meta.pop("eager_input_vals", None)

        # Replace all uses of the original node with the ordered version
        dependent_node.replace_all_uses_with(ordered_node)

        # Remove the original node from the graph
        graph.erase_node(dependent_node)

        # Now rename the ordered node to the original name
        ordered_node.name = original_name  # PRESERVE ORIGINAL NAME

        # Track the replacement for future dependencies
        replacements[dependent_node] = ordered_node


def _create_subgraph_for_node(graph: fx.Graph, node: fx.Node) -> fx.GraphModule:
    """
    Create a subgraph that exactly recreates a node's operation.

    The subgraph takes only the fx.Node arguments and recreates the operation
    with the exact target, args structure, and kwargs.

    Args:
        graph: The parent graph
        node: The node to wrap in a subgraph

    Returns:
        A GraphModule containing the subgraph
    """
    # Get the owning module
    # torch.distributed.breakpoint(0)
    owning_module = graph.owning_module

    # Create a new graph for the subgraph
    subgraph = fx.Graph(owning_module)

    new_args: list[Any] = []
    placeholder_idx = 0
    for _, arg in enumerate(node.args):
        if not isinstance(arg, fx.Node):
            new_args.append(arg)
            continue

        placeholder = subgraph.placeholder(f"arg_{placeholder_idx}")
        placeholder_idx += 1
        if "val" in arg.meta:
            placeholder.meta.update(arg.meta)
        new_args.append(placeholder)  # type: ignore[arg-type]

    new_kwargs: dict[str, Any] = {}
    for key, value in node.kwargs.items():
        if not isinstance(value, fx.Node):
            new_kwargs[key] = value
            continue

        placeholder = subgraph.placeholder(f"kwarg_{key}")
        if "val" in value.meta:
            placeholder.meta.update(value.meta)

        new_kwargs[key] = placeholder  # type: ignore[assignment]

    # Recreate the exact original operation in the subgraph
    assert callable(node.target)
    result = subgraph.call_function(
        node.target,
        tuple(new_args),
        new_kwargs,  # type: ignore[arg-type]
    )

    # Copy metadata from the original node
    result.meta.update(node.meta)

    out = subgraph.output(result)
    if "val" in result.meta:
        out.meta["val"] = result.meta["val"]

    return fx.GraphModule(owning_module, subgraph)
