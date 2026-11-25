# mypy: allow-untyped-defs
"""
Effect ordering pass for inductor.

This pass adds ordering dependencies to FX graphs using the control_deps HOP
for precise control over scheduling constraints. When you need exact ordering between
operations (e.g., collective_start -> mm -> wait), this pass wraps operations
with control_deps to make dependencies explicit.
"""

from operator import attrgetter, getitem
from typing import cast

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch.fx._lazy_graph_module import _LazyGraphModule
from torch.utils._ordered_set import OrderedSet


class ControlDeps(HigherOrderOperator):
    """
    Higher-order operator that enforces ordering by making dependencies explicit.

    Schema: control_deps(additional_deps, subgraph, *args, **kwargs) -> result
    where:
    - additional_deps: tuple of tensors that must be computed before this op
      (ordering-only, not a real data use)
    - subgraph: GraphModule containing the exact operation to execute
    - *args: pass-through arguments forwarded to the subgraph

    Semantics:
    - All tensors in additional_deps are computed before the subgraph executes.
    - Pass-through args (inputs returned unchanged by the subgraph) are
      versioned: future readers are ordered after all subgraph operations
      via a rename chain (OrderingOutput).  This ensures that consumers of
      a pass-through value cannot be scheduled before the subgraph's sync
      ops (e.g. wait_event) complete.
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
        # pyrefly: ignore [missing-attribute]
        return super().__call__(additional_deps, subgraph, *args, **kwargs)


control_deps = ControlDeps()

# control_deps wraps side-effecting ops (e.g. record_event, wait_event)
# and must not be eliminated by DCE even when its outputs are unused.
from torch.fx.node import has_side_effect


has_side_effect(control_deps)


# Register fake implementation for tracing
@register_fake(control_deps)
def _(additional_deps, subgraph, *args, **kwargs):
    """Fake tensor implementation - execute the subgraph."""
    return subgraph(*args, **kwargs)


# Register eager execution implementation
@control_deps.py_impl(DispatchKey.CompositeExplicitAutograd)
def control_deps_eager(additional_deps, subgraph, *args, **kwargs):
    """Eager implementation - just execute the subgraph."""
    return subgraph(*args, **kwargs)


# Autograd impl needed because additional_deps tensors may have autograd state,
# causing dispatch through AutogradCUDA even in post-autograd graphs.
@control_deps.py_impl(DispatchKey.Autograd)
def control_deps_autograd(additional_deps, subgraph, *args, **kwargs):
    return subgraph(*args, **kwargs)


def get_subgraph_name(gm: fx.GraphModule, name):
    name = f"subgraph_{name}"

    if not hasattr(gm, name):
        return name

    i = 0
    while hasattr(gm, f"{name}_{i}"):
        i += 1

    return f"{name}_{i}"


def _extract_unique_nodes(
    args: tuple[fx.node.Argument, ...], kwargs: dict[str, fx.node.Argument]
) -> tuple[list[fx.Node], list[fx.node.Argument], pytree.TreeSpec]:
    """Extract unique fx.Node instances from args/kwargs using pytree.

    Args:
        args: The positional arguments (may contain nested structures with fx.Node)
        kwargs: The keyword arguments (may contain nested structures with fx.Node)

    Returns:
        - Ordered list of unique fx.Node instances (preserves first occurrence order)
        - Flattened list of all items from args/kwargs
        - The pytree spec for reconstructing the original structure
    """
    flat_args_kwargs, spec = pytree.tree_flatten((args, kwargs))
    flat_args_kwargs = cast(list[fx.node.Argument], flat_args_kwargs)
    unique_nodes: list[fx.Node] = []
    seen: OrderedSet[fx.Node] = OrderedSet()
    for item in flat_args_kwargs:
        if isinstance(item, fx.Node) and item not in seen:
            unique_nodes.append(item)
            seen.add(item)
    return unique_nodes, flat_args_kwargs, spec


def preserve_node_ordering(
    graph: fx.Graph,
    additional_deps_map: dict[fx.Node, OrderedSet[fx.Node]],
    verbose: bool = False,
    *,
    mutation_deps: dict[fx.Node, OrderedSet[fx.Node]] | None = None,
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
        mutation_deps: Optional consumer-to-predecessor table for this graph.
            Keys and predecessors are remapped in place after wrapping.
    """
    if not additional_deps_map:
        return

    # Track replacements so we can update dependencies
    replacements: dict[fx.Node, fx.Node] = {}

    # Process each node that needs additional dependencies
    for dependent_node, dep_nodes in additional_deps_map.items():
        if dependent_node.op != "call_function":
            raise AssertionError(dependent_node.op)

        original_name = dependent_node.name
        original_args = dependent_node.args
        original_kwargs = dependent_node.kwargs
        original_meta = dependent_node.meta.copy()

        updated_dep_nodes = [replacements.get(dep, dep) for dep in dep_nodes]

        # Create a subgraph that preserves the exact original operation
        subgraph_module = _create_subgraph_for_node(graph, dependent_node)

        owning_mod = graph.owning_module
        if owning_mod is None:
            raise AssertionError("expected graph to have an owning_module")
        subgraph_attr_name = get_subgraph_name(owning_mod, original_name)
        setattr(graph.owning_module, subgraph_attr_name, subgraph_module)

        # Create control_deps call with:
        # 1. Additional dependencies as first arg (explicit)
        # 2. Subgraph via get_attr (like b2b gemm pass)
        # 3. Original arguments (only fx.Node args and kwargs are passed)
        with graph.inserting_before(dependent_node):
            # Create get_attr node for the subgraph
            get_subgraph = graph.get_attr(subgraph_attr_name)

            # Extract unique nodes from nested args/kwargs
            node_args, _, _ = _extract_unique_nodes(original_args, original_kwargs)

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
        # The wrapped operation retains its fallback metadata in the subgraph.
        # The control_deps HOP itself must use its dedicated lowering because
        # FallbackKernel cannot handle its Subgraph argument.
        ordered_node.meta.pop("should_fallback", None)
        custom_meta = ordered_node.meta.get("custom")
        if isinstance(custom_meta, dict) and "fallback_to_eager" in custom_meta:
            ordered_node.meta["custom"] = custom_meta.copy()
            ordered_node.meta["custom"].pop("fallback_to_eager")

        # Replace all uses of the original node with the ordered version
        dependent_node.replace_all_uses_with(ordered_node)

        # Remove the original node from the graph
        graph.erase_node(dependent_node)

        # Now rename the ordered node to the original name
        ordered_node.name = original_name  # PRESERVE ORIGINAL NAME

        # Track the replacement for future dependencies
        replacements[dependent_node] = ordered_node

    if mutation_deps is not None:
        updated = {
            replacements.get(n, n): OrderedSet(replacements.get(d, d) for d in deps)
            for n, deps in mutation_deps.items()
        }
        mutation_deps.clear()
        mutation_deps.update(updated)


def _get_subgraph(reference: object) -> fx.GraphModule | None:
    if (
        not isinstance(reference, fx.Node)
        or reference.op != "get_attr"
        or not isinstance(reference.target, str)
    ):
        return None
    subgraph = reference.graph.owning_module
    if subgraph is None:
        return None
    for name in reference.target.split("."):
        if not hasattr(subgraph, name):
            return None
        subgraph = getattr(subgraph, name)
    return subgraph if isinstance(subgraph, fx.GraphModule) else None


def _node_storages(node: fx.Node) -> OrderedSet[int] | None:
    from torch._inductor.fx_utils import get_node_storage, get_storage

    if node.meta.get("val") is None:
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and not node.target._schema.returns
        ):
            return OrderedSet()
        if node.target is torch.ops.higher_order.triton_kernel_wrapper_mutation:
            return OrderedSet()
        if node.target is control_deps:
            subgraph = _get_subgraph(node.args[1])
            if subgraph is None:
                return None
            output = next(n for n in subgraph.graph.nodes if n.op == "output")
            storages: OrderedSet[int] = OrderedSet()
            for source in pytree.tree_leaves(output.args[0]):
                if isinstance(source, fx.Node):
                    source_storages = _node_storages(source)
                    if source_storages is None:
                        return None
                    storages.update(source_storages)
            return storages
        if _get_subgraph(node) is not None:
            return OrderedSet()
        return None
    storage = get_node_storage(node)
    if storage is not None:
        return OrderedSet([storage])
    storages: OrderedSet[int] = OrderedSet()
    for value in pytree.tree_leaves(node.meta["val"]):
        if isinstance(value, torch.Tensor):
            if not torch._C._has_storage(value):
                return None
            storages.add(get_storage(value))
    return storages


def _get_written_storages(node: fx.Node) -> OrderedSet[int] | None:
    r"""Find storages written by this node or its wrapped subgraph.

    Return ``None`` when required storage metadata is unavailable, and an
    empty set when the node has no writes.

    Example::
        INPUT:
            x = placeholder()           # storage 1
            y = placeholder()           # storage 2
            copied = aten.copy_(x, y)   # storage 1
            result = output(copied)

            node = copied

        OUTPUT:
            OrderedSet([1])
    """
    from torch._inductor.fx_utils import get_mutated_input_nodes

    storages: OrderedSet[int] = OrderedSet()
    if node.target is control_deps:
        subgraph = _get_subgraph(node.args[1])
        if subgraph is None:
            return None
        nested_writes = _graph_storage_writes(subgraph)
        if nested_writes is None:
            return None
        for nested in nested_writes.values():
            storages.update(nested)
    for written in get_mutated_input_nodes(node):
        value = written.meta.get("val")
        if not any(isinstance(x, torch.Tensor) for x in pytree.tree_leaves(value)):
            target = (
                node.args[1]
                if node.target is torch.ops.higher_order.with_effects
                else node.target
            )
            if (
                target is torch.ops.higher_order.triton_kernel_wrapper_mutation
                and isinstance(
                    value,
                    (
                        int,
                        float,
                        bool,
                        complex,
                        torch.SymInt,
                        torch.SymFloat,
                        torch.SymBool,
                    ),
                )
            ):
                continue
            return None
        input_storages = _node_storages(written)
        if input_storages is None:
            return None
        storages.update(input_storages)
    return storages


def _graph_storage_writes(
    gm: fx.GraphModule,
) -> dict[fx.Node, OrderedSet[int]] | None:
    from torch._inductor.fx_passes.bucketing import _get_collective_node_from_wait

    writes: dict[fx.Node, OrderedSet[int]] = {}
    for node in gm.graph.nodes:
        storages = _get_written_storages(node)
        if storages is None:
            return None
        collective = _get_collective_node_from_wait(node)
        if collective is not None:
            # An out collective's write completes at its wait.
            storages.update(writes.get(collective, ()))
        if storages:
            writes[node] = storages
    return writes


def _storage_reads(node: fx.Node, has_writes: bool) -> OrderedSet[int] | None:
    from torch._inductor.fx_passes.bucketing import _get_collective_node_from_wait
    from torch._inductor.fx_passes.fusion_regions import is_view_node

    first = node.args[0] if node.args else None
    value = first.meta.get("val") if isinstance(first, fx.Node) else first
    if not has_writes and node.target is getitem and type(value) in (tuple, list, dict):
        return OrderedSet()
    reads: OrderedSet[int] = OrderedSet()
    sources = list(node.all_input_nodes)
    collective = _get_collective_node_from_wait(node)
    if collective is not None:
        # NCCL can access the launch inputs until its wait completes.
        sources.extend(collective.all_input_nodes)
    for source in sources:
        storages = _node_storages(source)
        if storages is None:
            return None
        reads.update(storages)
    if not has_writes and is_view_node(node):
        outputs = _node_storages(node)
        if outputs and outputs <= reads:
            return OrderedSet()
    return reads


def _mutation_ordering_plan(
    gm: fx.GraphModule, writes: dict[fx.Node, OrderedSet[int]]
) -> dict[fx.Node, OrderedSet[fx.Node]] | None:
    mutated = OrderedSet(s for storages in writes.values() for s in storages)
    nodes = list(gm.graph.nodes)
    node_indices = {node: index for index, node in enumerate(nodes)}
    ancestors: dict[fx.Node, int] = {}
    last_writer: dict[int, fx.Node] = {}
    readers: dict[int, OrderedSet[fx.Node]] = {}
    plan: dict[fx.Node, OrderedSet[fx.Node]] = {}
    for node in nodes:
        covered = 0
        for source in node.all_input_nodes:
            covered |= ancestors[source] | (1 << node_indices[source])
        written = writes.get(node, OrderedSet())
        reads = _storage_reads(node, bool(written))
        if reads is None:
            return None
        reads &= mutated
        deps: OrderedSet[fx.Node] = OrderedSet()
        for storage in reads | written:
            if storage in last_writer:
                deps.add(last_writer[storage])
        for storage in written:
            deps.update(readers.get(storage, ()))
        for dep in deps:
            covered |= ancestors[dep]
        deps = OrderedSet(dep for dep in deps if not (covered >> node_indices[dep]) & 1)
        ancestors[node] = covered
        for dep in deps:
            ancestors[node] |= 1 << node_indices[dep]
        if deps:
            plan[node] = deps
        for storage in reads - written:
            readers.setdefault(storage, OrderedSet()).add(node)
        for storage in written:
            last_writer[storage] = node
            readers[storage] = OrderedSet()
    return plan


def get_mutation_dependencies(
    gm: fx.GraphModule,
) -> dict[fx.Node, OrderedSet[fx.Node]] | None:
    r"""Collect storage RAW, WAR, and WAW dependencies without changing the graph.

    Return ``None`` when required storage metadata is unavailable.
    """
    writes = _graph_storage_writes(gm)
    if writes is None:
        return None
    if not writes:
        return {}
    return _mutation_ordering_plan(gm, writes)


def _create_subgraph_for_node(
    graph: fx.Graph, node: fx.Node, additional_deps=None
) -> fx.GraphModule:
    """
    Create a subgraph that exactly recreates a node's operation optionally passing through additional dependencies.

    The subgraph takes only the fx.Node arguments and recreates the operation
    with the exact target, args structure, and kwargs.

    Args:
        graph: The parent graph
        node: The node to wrap in a subgraph
        additional_deps: Additional dependencies to pass through the subgraph

    Returns:
        A GraphModule containing the subgraph
    """
    # Get the owning module
    owning_module = graph.owning_module
    if owning_module is None:
        raise AssertionError("graph.owning_module must not be None")

    # Create a new graph for the subgraph
    subgraph = fx.Graph(owning_module)

    # Extract unique nodes and get flattened structure + spec
    unique_nodes, flat_args_kwargs, spec = _extract_unique_nodes(node.args, node.kwargs)

    # Create placeholders for each unique node
    node_to_placeholder: dict[fx.Node, fx.Node] = {}
    for idx, orig_node in enumerate(unique_nodes):
        placeholder = subgraph.placeholder(f"arg_{idx}")
        if "val" in orig_node.meta:
            placeholder.meta.update(orig_node.meta)
        elif orig_node.op == "get_attr" and isinstance(orig_node.target, str):
            placeholder.meta["val"] = attrgetter(orig_node.target)(owning_module)
        node_to_placeholder[orig_node] = placeholder

    # Replace fx.Node instances with their placeholders
    def replace_nodes(item: fx.node.Argument) -> fx.node.Argument:
        if isinstance(item, fx.Node):
            return node_to_placeholder[item]
        return item

    additional_deps_placeholders = []
    for idx, dep in enumerate(additional_deps or ()):
        placeholder = subgraph.placeholder(f"dep_{idx}")
        if "val" in dep.meta:
            placeholder.meta.update(dep.meta)
        additional_deps_placeholders.append(placeholder)

    new_flat = [replace_nodes(item) for item in flat_args_kwargs]
    new_args, new_kwargs = cast(
        tuple[tuple[fx.node.Argument, ...], dict[str, fx.node.Argument]],
        pytree.tree_unflatten(new_flat, spec),
    )

    # Recreate the exact original operation in the subgraph
    if not callable(node.target):
        raise AssertionError(f"expected node.target to be callable, got {node.target}")
    result = subgraph.call_function(
        node.target,
        tuple(new_args),
        new_kwargs,
    )

    # Copy metadata from the original node
    result.meta.update(node.meta)

    if additional_deps_placeholders:
        outputs = tuple([result] + additional_deps_placeholders)
        out = subgraph.output(outputs)
        out.meta["val"] = tuple(output.meta.get("val") for output in outputs)
    else:
        out = subgraph.output(result)
        if "val" in result.meta:
            out.meta["val"] = result.meta["val"]

    return _LazyGraphModule(owning_module, subgraph)
