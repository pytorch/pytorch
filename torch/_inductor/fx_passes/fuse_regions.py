"""FX helper for carrying Inductor fusion-region boundaries.

``mark_fuse_region`` outlines a set of operation nodes into an
``invoke_subgraph`` HOP and stores the region id in ``node.meta[FUSE_REGION]``.
Inductor lowering consumes that metadata, inlines the subgraph, and copies the
same id onto the generated IR operation annotations. The scheduler then treats
different ids as fusion and non-adjacent buffer-reuse barriers.
"""

# mypy: allow-untyped-defs
from operator import attrgetter, getitem
from typing import Any

import torch
import torch.fx as fx
from torch.fx._lazy_graph_module import _LazyGraphModule
from torch.utils._ordered_set import OrderedSet


FUSE_REGION = "fuse_region"


def _get_subgraph_name(gm: fx.GraphModule, name: str) -> str:
    i = 0
    while hasattr(gm, f"{name}_{i}"):
        i += 1
    return f"{name}_{i}"


def _copy_placeholder_meta(
    placeholder: fx.Node, input_node: fx.Node, owning_module: fx.GraphModule
) -> None:
    if "val" in input_node.meta:
        placeholder.meta.update(input_node.meta)
    elif input_node.op == "get_attr" and isinstance(input_node.target, str):
        placeholder.meta["val"] = attrgetter(input_node.target)(owning_module)


def _getattr_or_none(module: fx.GraphModule, target: str) -> Any:
    try:
        return attrgetter(target)(module)
    except AttributeError:
        return None


def _has_graph_module_arg(node: fx.Node) -> bool:
    gm = node.graph.owning_module
    if gm is None:
        return False
    return any(
        inp.op == "get_attr"
        and isinstance(inp.target, str)
        and isinstance(_getattr_or_none(gm, inp.target), fx.GraphModule)
        for inp in node.all_input_nodes
    )


def fuse_region_key(node: fx.Node) -> str | None:
    if node.op in ("placeholder", "output", "get_attr"):
        return None
    if node.op == "call_function" and isinstance(
        node.target, torch._ops.HigherOrderOperator
    ):
        return None
    if _has_graph_module_arg(node):
        return None

    custom = node.meta.get("custom")
    if not isinstance(custom, dict):
        return None
    region = _fuse_region_from_custom(custom)
    if region is None:
        return None
    phase = "bwd" if node.meta.get("autograd_backward") is True else "fwd"
    return f"{region}_{phase}"


def _fuse_region_from_custom(custom: dict[str, Any]) -> str | None:
    region = custom.get(FUSE_REGION)
    if region is not None:
        if not isinstance(region, str):
            raise AssertionError(
                f"expected custom {FUSE_REGION} to be a str, got {type(region)}"
            )
        return region

    compile_with_inductor = custom.get("compile_with_inductor")
    if isinstance(compile_with_inductor, dict):
        region = compile_with_inductor.get(FUSE_REGION)
        if region is not None:
            if not isinstance(region, str):
                raise AssertionError(
                    f"expected custom compile_with_inductor {FUSE_REGION} "
                    f"to be a str, got {type(region)}"
                )
            return region
    return None


def _strip_fuse_region_from_meta(meta: dict[str, Any]) -> None:
    custom = meta.get("custom")
    if not isinstance(custom, dict):
        return

    custom = custom.copy()
    custom.pop(FUSE_REGION, None)
    compile_with_inductor = custom.get("compile_with_inductor")
    if isinstance(compile_with_inductor, dict):
        compile_with_inductor = compile_with_inductor.copy()
        compile_with_inductor.pop(FUSE_REGION, None)
        custom["compile_with_inductor"] = compile_with_inductor

    if custom:
        meta["custom"] = custom
    else:
        meta.pop("custom", None)


def collect_fuse_region_groups(graph: fx.Graph) -> list[tuple[str, list[fx.Node]]]:
    groups: list[tuple[str, list[fx.Node]]] = []
    current_key: str | None = None
    current_nodes: list[fx.Node] = []

    def flush() -> None:
        nonlocal current_key, current_nodes
        if current_key is not None and len(current_nodes) > 1:
            groups.append((current_key, current_nodes))
        current_key = None
        current_nodes = []

    for node in list(graph.nodes):
        key = fuse_region_key(node)
        if key is None:
            flush()
            continue
        if key != current_key:
            flush()
            current_key = key
        current_nodes.append(node)
    flush()
    return groups


def fuse_region_annotations_cache_key(
    gm: fx.GraphModule,
) -> tuple[tuple[str, tuple[tuple[str, tuple[str, ...]], ...]], ...]:
    entries = []
    for module_name, module in gm.named_modules():
        if not isinstance(module, fx.GraphModule):
            continue
        groups = collect_fuse_region_groups(module.graph)
        if groups:
            group_entries = tuple(
                (key, tuple(node.name for node in nodes)) for key, nodes in groups
            )
            entries.append((module_name, group_entries))
    return tuple(entries)


def apply_fuse_region_annotations(graph: fx.Graph) -> None:
    """
    Outline contiguous FX nodes annotated with ``node.meta["custom"][FUSE_REGION]``.

    ``torch.fx.traceback.annotate`` stores user metadata in ``node.meta["custom"]``.
    This pass consumes string-valued ``FUSE_REGION`` annotations from that dict,
    or from the nested ``compile_with_inductor`` annotation used by
    regional_inductor callers, and turns each contiguous same-id run into one
    invoke_subgraph fuse region.
    """
    groups = collect_fuse_region_groups(graph)
    if not groups:
        return
    for key, nodes in groups:
        mark_fuse_region(graph, nodes, fuse_region_id=key)
    graph.lint()


def mark_fuse_region(
    graph: fx.Graph,
    nodes: list[fx.Node],
    fuse_region_id: str | None = None,
) -> fx.Node | tuple[fx.Node, ...]:
    """
    Outline FX nodes into an invoke_subgraph fusion region.

    Inductor lowers the invoke_subgraph inline and annotates the produced IR
    operations with one fusion-region id. Operations inside the region may fuse
    with each other, but not with operations outside the region or with another
    region id.
    """
    owning_module = graph.owning_module
    if owning_module is None:
        raise AssertionError("expected graph to have an owning_module")
    if not nodes:
        raise AssertionError("expected non-empty nodes")
    if fuse_region_id is not None and not isinstance(fuse_region_id, str):
        raise AssertionError(
            f"expected fuse_region_id to be None or str, got {type(fuse_region_id)}"
        )

    node_set = OrderedSet(nodes)
    ordered_nodes = [node for node in graph.nodes if node in node_set]
    if len(ordered_nodes) != len(node_set):
        raise AssertionError("expected all nodes to belong to graph")
    if any(node.op in ("placeholder", "output") for node in ordered_nodes):
        raise AssertionError("expected fuse_region nodes to exclude graph boundaries")

    region_outputs = [
        node
        for node in ordered_nodes
        if any(user not in node_set for user in node.users)
    ]

    subgraph = fx.Graph(owning_module)
    env: dict[fx.Node, fx.Node] = {}
    input_replacements: dict[fx.Node, Any] = {}
    boundary_args: list[tuple[fx.Node, tuple[int, ...], Any]] = []

    external_inputs: OrderedSet[fx.Node] = OrderedSet()
    preserved_getattrs: OrderedSet[fx.Node] = OrderedSet()

    def collect_external_input(node: fx.Node) -> fx.Node:
        if node not in node_set:
            if (
                node.op == "get_attr"
                and isinstance(node.target, str)
                and isinstance(attrgetter(node.target)(owning_module), fx.GraphModule)
            ):
                preserved_getattrs.add(node)
            else:
                external_inputs.add(node)
        return node

    for node in ordered_nodes:
        fx.map_arg((node.args, node.kwargs), collect_external_input)

    node_order = {node: idx for idx, node in enumerate(graph.nodes)}
    latest_input = max(
        external_inputs,
        key=lambda node: node_order[node],
        default=None,
    )
    first_external_user = min(
        (
            user
            for output_node in region_outputs
            for user in output_node.users
            if user not in node_set
        ),
        key=lambda node: node_order[node],
        default=None,
    )
    if (
        first_external_user is not None
        and latest_input is not None
        and node_order[latest_input] >= node_order[first_external_user]
    ):
        raise AssertionError("expected fuse_region boundary to be acyclic")

    def add_boundary_arg(
        input_node: fx.Node, path: tuple[int, ...], meta_val: Any
    ) -> fx.Node:
        placeholder = subgraph.placeholder(f"arg_{len(boundary_args)}")
        _copy_placeholder_meta(placeholder, input_node, owning_module)
        if path:
            placeholder.meta["val"] = meta_val
        boundary_args.append((input_node, path, meta_val))
        return placeholder

    def make_input_replacement(
        input_node: fx.Node, value: Any, path: tuple[int, ...] = ()
    ) -> Any:
        if isinstance(value, (tuple, list)):
            return type(value)(
                make_input_replacement(input_node, item, (*path, idx))
                for idx, item in enumerate(value)
            )
        return add_boundary_arg(input_node, path, value)

    for input_node in external_inputs:
        value = input_node.meta.get("val")
        if isinstance(value, (tuple, list)):
            input_replacements[input_node] = make_input_replacement(input_node, value)
        else:
            input_replacements[input_node] = add_boundary_arg(input_node, (), value)

    def load_arg(node: fx.Node) -> Any:
        if node in env:
            return env[node]
        if node in node_set:
            raise AssertionError("expected fuse_region nodes to be topological")
        if node in preserved_getattrs:
            if not isinstance(node.target, str):
                raise AssertionError("expected get_attr target to be a string")
            get_attr_node = subgraph.get_attr(node.target)
            get_attr_node.meta.update(node.meta)
            env[node] = get_attr_node
            return get_attr_node
        return input_replacements[node]

    for node in ordered_nodes:
        env[node] = subgraph.node_copy(node, load_arg)

    subgraph_outputs = tuple(env[node] for node in region_outputs)
    if len(subgraph_outputs) == 0:
        out = subgraph.output(())
        out.meta["val"] = ()
    elif len(subgraph_outputs) == 1:
        out = subgraph.output(subgraph_outputs[0])
        if "val" in region_outputs[0].meta:
            out.meta["val"] = region_outputs[0].meta["val"]
    else:
        out = subgraph.output(subgraph_outputs)
        out.meta["val"] = tuple(node.meta.get("val") for node in region_outputs)
    subgraph.lint()

    subgraph_module = _LazyGraphModule(owning_module, subgraph)
    region_name = f"fuse_region_{ordered_nodes[0].name}_{ordered_nodes[-1].name}"
    subgraph_attr_name = _get_subgraph_name(owning_module, region_name)
    setattr(owning_module, subgraph_attr_name, subgraph_module)

    if latest_input is None or node_order[latest_input] < node_order[ordered_nodes[0]]:
        with graph.inserting_before(ordered_nodes[0]):
            get_subgraph = graph.get_attr(subgraph_attr_name)
    else:
        with graph.inserting_after(latest_input):
            get_subgraph = graph.get_attr(subgraph_attr_name)

    outer_args: list[fx.Node] = []
    insert_after = get_subgraph

    def make_outer_arg(
        input_node: fx.Node, path: tuple[int, ...], meta_val: Any
    ) -> fx.Node:
        nonlocal insert_after
        source = input_node
        for idx in path:
            with graph.inserting_after(insert_after):
                source = graph.call_function(
                    getitem,
                    args=(source, idx),
                    name=f"{input_node.name}_fuse_region_arg_{len(outer_args)}",
                )
            insert_after = source
        if path:
            source.meta["val"] = meta_val
        return source

    for input_node, path, meta_val in boundary_args:
        outer_args.append(make_outer_arg(input_node, path, meta_val))

    with graph.inserting_after(insert_after):
        region_node = graph.call_function(
            torch.ops.higher_order.invoke_subgraph,
            args=(get_subgraph, subgraph_attr_name, *outer_args),
            name=region_name,
        )

    replacements: list[fx.Node] = []
    if len(region_outputs) == 0:
        region_node.meta["val"] = ()
    elif len(region_outputs) == 1:
        replacement = region_node
        replacement.meta = region_outputs[0].meta.copy()
        replacement.meta.pop("eager_input_vals", None)
        _strip_fuse_region_from_meta(replacement.meta)
        replacements.append(replacement)
    else:
        region_node.meta["val"] = tuple(node.meta.get("val") for node in region_outputs)
        insert_after = region_node
        for idx, output_node in enumerate(region_outputs):
            with graph.inserting_after(insert_after):
                replacement = graph.call_function(
                    getitem,
                    args=(region_node, idx),
                    name=f"{output_node.name}_fuse_region",
                )
            replacement.meta = output_node.meta.copy()
            replacement.meta.pop("eager_input_vals", None)
            _strip_fuse_region_from_meta(replacement.meta)
            replacements.append(replacement)
            insert_after = replacement

    for output_node, replacement in zip(region_outputs, replacements, strict=True):
        for user in list(output_node.users):
            if user not in node_set:
                user.replace_input_with(output_node, replacement)

    for node in reversed(ordered_nodes):
        graph.erase_node(node)
    graph.lint()

    region_node.meta[FUSE_REGION] = (
        region_name if fuse_region_id is None else fuse_region_id
    )

    if not replacements:
        return region_node
    return replacements[0] if len(replacements) == 1 else tuple(replacements)
