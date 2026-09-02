# mypy: allow-untyped-defs
"""
Optimize complete chains that assemble a tensor from slices.

Match and replace contiguous chunks as follows:

    out = base
    copied0 = aten.copy(aten.slice(out, dim, 0, end0), chunk0)
    out = aten.slice_scatter(out, copied0, dim, 0, end0)
    copied1 = aten.copy(aten.slice(out, dim, end0, size), chunk1)
    out = aten.slice_scatter(out, copied1, dim, end0, size)

    aten.copy_(aten.slice(base, dim, 0, end0), chunk0)
    aten.copy_(aten.slice(base, dim, end0, size), chunk1)
    out = base

The source may be the chunk directly, or functionalization may wrap it in a
fully overwriting ``copy(slice(out), chunk)`` and insert aliases around
``out``. Reuse an ``empty`` or ``empty_strided`` base allocation and replace
the chain with ``copy_`` into its slices. This consumes each chunk where its
original ``slice_scatter`` occurred instead of keeping every chunk live until
the end of the chain. Canonical single-use graph inputs, non-realized contiguous
pointwise chunks, and CUDA outputs of nested compile regions instead become
``cat`` so they can use one output copy.

The writes must replace every value from ``base``, and the output and chunks
must have the same dtype and device. Other chains are left unchanged for normal
lowering.
"""

import operator
from dataclasses import dataclass
from typing import Any

import torch
from torch._inductor.fx_utils import is_node_realized
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    compute_mutation_region_ids,
    Match,
    MULTIPLE,
)
from torch._prims_common import make_contiguous_strides_for
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq
from torch.fx.operator_schemas import normalize_function
from torch.fx.passes.reinplace import _is_view_op
from torch.utils._ordered_set import OrderedSet


aten = torch.ops.aten
_SLICE_SCATTER_PATTERN = CallFunctionVarArgs(aten.slice_scatter.default, users=MULTIPLE)
_SUPPORTED_BASE_FACTORIES = (
    aten.empty.memory_format,
    aten.empty_strided.default,
)


@dataclass
class _SliceScatterChain:
    base: torch.fx.Node
    slice_scatters: list[torch.fx.Node]
    dim: int
    sources: list[torch.fx.Node]
    removable_nodes: OrderedSet[torch.fx.Node]
    functional_copies: list[torch.fx.Node]
    non_blocking: list[Any]


def _normalize_slice_scatter_args(node: torch.fx.Node) -> dict[str, Any]:
    normalized = normalize_function(
        aten.slice_scatter.default,
        args=node.args,
        kwargs=node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if normalized is None:
        raise AssertionError(f"failed to normalize arguments for {node}")
    return normalized.kwargs


def _get_val(value: Any) -> Any:
    return value.meta.get("val") if isinstance(value, torch.fx.Node) else value


def _statically_known_eq(lhs: Any, rhs: Any) -> bool:
    lhs = _get_val(lhs)
    rhs = _get_val(rhs)
    if lhs is None or rhs is None:
        return False
    return statically_known_true(sym_eq(lhs, rhs))


def _statically_known_same_shape(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return lhs.dim() == rhs.dim() and all(
        _statically_known_eq(lhs_size, rhs_size)
        for lhs_size, rhs_size in zip(lhs.shape, rhs.shape, strict=True)
    )


def _custom_context(node: torch.fx.Node) -> tuple[Any, Any, Any]:
    custom = node.meta.get("custom", {})
    return (
        custom.get("stream", 0),
        custom.get("mempool"),
        custom.get("mempool_device"),
    )


def _strip_aliases(
    node: Any,
    removable_nodes: OrderedSet[torch.fx.Node],
    examined: OrderedSet[torch.fx.Node],
) -> Any:
    while isinstance(node, torch.fx.Node) and node.target is aten.alias.default:
        if node in examined:
            return None
        examined.add(node)
        removable_nodes.add(node)
        node = node.args[0]
    return node


def _copy_source(
    node: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node, Any] | None:
    """Return the payload of a non-broadcasting, non-converting functional copy.

    A functional copy completely overwrites its first argument. When its payload
    already has the output shape, dtype, and device, the result has the same
    values as the payload without reading the destination.
    """
    normalized = normalize_function(
        aten.copy.default,
        args=node.args,
        kwargs=node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if normalized is None:
        raise AssertionError(f"failed to normalize arguments for {node}")
    copy_input = normalized.kwargs["input"]
    payload = normalized.kwargs["src"]
    result_val = node.meta.get("val")
    payload_val = (
        payload.meta.get("val") if isinstance(payload, torch.fx.Node) else None
    )
    if (
        not isinstance(payload, torch.fx.Node)
        or not isinstance(copy_input, torch.fx.Node)
        or not isinstance(result_val, torch.Tensor)
        or not isinstance(payload_val, torch.Tensor)
        or result_val.dtype != payload_val.dtype
        or result_val.device != payload_val.device
        or not _statically_known_same_shape(result_val, payload_val)
    ):
        return None
    return payload, copy_input, normalized.kwargs["non_blocking"]


def _view_path_to_state(
    node: Any,
    state_nodes: OrderedSet[torch.fx.Node],
    examined: OrderedSet[torch.fx.Node],
) -> list[torch.fx.Node] | None:
    path = []
    while isinstance(node, torch.fx.Node) and _is_view_op(node.target):
        if node in examined:
            return None
        examined.add(node)
        path.append(node)
        node = node.args[0]
    return path if node in state_nodes else None


def _collect_dead_view_nodes(
    node: torch.fx.Node, removable_nodes: OrderedSet[torch.fx.Node]
) -> bool:
    postorder = []
    is_view = {}
    seen = OrderedSet[torch.fx.Node]()
    stack = [(node, False)]
    while stack:
        current, expanded = stack.pop()
        if current in removable_nodes:
            continue
        if expanded:
            postorder.append(current)
            continue
        if current in seen:
            continue
        seen.add(current)
        current_is_view = _is_view_op(current.target)
        is_view[current] = current_is_view
        stack.append((current, True))
        if current_is_view:
            stack.extend((user, False) for user in current.users)

    for current in postorder:
        if is_view[current] and all(user in removable_nodes for user in current.users):
            removable_nodes.add(current)
    return node in removable_nodes


def _get_full_tensor_slice_scatter_chain(
    slice_scatter: torch.fx.Node,
    examined: OrderedSet[torch.fx.Node],
) -> _SliceScatterChain | None:
    """Return a chain whose adjacent slices overwrite the entire base tensor."""
    removable_nodes = OrderedSet[torch.fx.Node]()
    slice_scatters = []
    current = slice_scatter
    while (
        isinstance(current, torch.fx.Node)
        and current.target is aten.slice_scatter.default
    ):
        if current in examined:
            return None
        examined.add(current)
        slice_scatters.append(current)
        removable_nodes.add(current)
        current = _strip_aliases(
            _normalize_slice_scatter_args(current)["input"], removable_nodes, examined
        )
    slice_scatters.reverse()

    if not isinstance(current, torch.fx.Node) or len(slice_scatters) < 2:
        return None

    base_val = current.meta.get("val")
    if (
        not isinstance(base_val, torch.Tensor)
        or base_val.dim() == 0
        or torch._debug_has_internal_overlap(base_val) != 0
    ):
        return None

    dim = _normalize_slice_scatter_args(slice_scatters[0])["dim"]
    if not isinstance(dim, int):
        return None
    dim %= base_val.dim()

    cursor = 0
    sources = []
    functional_copies = []
    non_blocking = []
    state_nodes = OrderedSet([current, *slice_scatters[:-1]])
    copy_view_nodes = OrderedSet[torch.fx.Node]()
    for matched_slice_scatter in slice_scatters:
        args = _normalize_slice_scatter_args(matched_slice_scatter)
        node_dim = args["dim"]
        start = args["start"]
        end = args["end"]
        step = args["step"]
        src = args["src"]
        if isinstance(src, torch.fx.Node) and src.target is aten.copy.default:
            copy_source = _copy_source(src)
            if copy_source is None:
                return None
            src, copy_input, copy_non_blocking = copy_source
            functional_copies.append(args["src"])
            removable_nodes.add(args["src"])
            view_path = _view_path_to_state(copy_input, state_nodes, copy_view_nodes)
            if view_path is None:
                return None
            removable_nodes.update(view_path)
        else:
            copy_non_blocking = False
        start_val = _get_val(start)
        end_val = _get_val(end)
        if not isinstance(node_dim, int) or not isinstance(src, torch.fx.Node):
            return None
        node_dim %= base_val.dim()
        src_val = src.meta.get("val")
        if (
            node_dim != dim
            or start_val is None
            or end_val is None
            or step != 1
            or not _statically_known_eq(start_val, cursor)
            or not isinstance(src_val, torch.Tensor)
            or src_val.dim() != base_val.dim()
            or src_val.dtype != base_val.dtype
            or src_val.device != base_val.device
            or any(
                not _statically_known_eq(src_size, base_size)
                for index, (src_size, base_size) in enumerate(
                    zip(src_val.shape, base_val.shape, strict=True)
                )
                if index != dim
            )
            or not _statically_known_eq(src_val.shape[dim], end_val - start_val)
        ):
            return None
        cursor = end_val
        sources.append(src)
        non_blocking.append(copy_non_blocking)

    if not _statically_known_eq(cursor, base_val.shape[dim]):
        return None

    for node in [current, *removable_nodes]:
        for user in list(node.users):
            _collect_dead_view_nodes(user, removable_nodes)

    if any(src is current or src in removable_nodes for src in sources) or any(
        node is not slice_scatter
        and any(user not in removable_nodes for user in node.users)
        for node in [current, *removable_nodes]
    ):
        return None

    custom_context = _custom_context(slice_scatter)
    mutation_region = slice_scatter.meta.get("mutation_region_id")
    if any(
        _custom_context(node) != custom_context
        or node.meta.get("mutation_region_id") != mutation_region
        for node in [*slice_scatters[:-1], *functional_copies]
    ):
        return None

    return _SliceScatterChain(
        current,
        slice_scatters,
        dim,
        sources,
        removable_nodes,
        functional_copies,
        non_blocking,
    )


def _is_supported_base_factory(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in _SUPPORTED_BASE_FACTORIES


def _is_reusable_base_factory(node: torch.fx.Node) -> bool:
    if not _is_supported_base_factory(node):
        return False
    target = node.target
    if not callable(target):
        return False
    normalized = normalize_function(
        target,
        args=node.args,
        kwargs=node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if normalized is None:
        return False
    pin_memory = normalized.kwargs.get("pin_memory")
    return pin_memory is None or pin_memory is False


def _replace_with_inplace_copies(match: Match, chain: _SliceScatterChain) -> None:
    graph = chain.base.graph
    for slice_scatter, source, non_blocking in zip(
        chain.slice_scatters, chain.sources, chain.non_blocking, strict=True
    ):
        args = _normalize_slice_scatter_args(slice_scatter)
        with graph.inserting_before(slice_scatter):
            dst = graph.call_function(
                aten.slice.Tensor,
                args=(
                    chain.base,
                    chain.dim,
                    args["start"],
                    args["end"],
                    args["step"],
                ),
            )
            copy = graph.call_function(
                aten.copy_.default, args=(dst, source, non_blocking)
            )
        if custom := slice_scatter.meta.get("custom"):
            dst.meta["custom"] = custom.copy()
            copy.meta["custom"] = custom.copy()

    chain.slice_scatters[-1].replace_all_uses_with(chain.base)
    match.erase_nodes()


def _is_nested_compile_region_output(node: torch.fx.Node) -> bool:
    if node.op != "call_function" or node.target is not operator.getitem:
        return False
    region = node.args[0]
    value = node.meta.get("val")
    return (
        isinstance(region, torch.fx.Node)
        and region.op == "call_function"
        and region.target is torch.ops.higher_order.invoke_subgraph
        and isinstance(value, torch.Tensor)
        and value.device.type == "cuda"
    )


def _can_fuse_as_cat(chain: _SliceScatterChain) -> bool:
    base_val = chain.base.meta.get("val")
    if (
        chain.functional_copies
        or not isinstance(base_val, torch.Tensor)
        or base_val.layout != torch.strided
    ):
        return False

    for src in chain.sources:
        is_input = src.op == "placeholder"
        is_pointwise = (
            isinstance(src.target, torch._ops.OpOverload)
            and torch.Tag.pointwise in src.target.tags
        )
        if (
            len(src.users) != 1
            or not (is_input or is_pointwise or _is_nested_compile_region_output(src))
            or (not is_input and is_node_realized(src))
        ):
            return False

    tensors = [base_val, *(src.meta.get("val") for src in chain.sources)]
    return all(
        isinstance(tensor, torch.Tensor)
        and tensor.device == base_val.device
        and tensor.dtype == base_val.dtype
        and tensor.layout == base_val.layout
        and all(
            _statically_known_eq(actual, expected)
            for actual, expected in zip(
                tensor.stride(),
                make_contiguous_strides_for(tensor.shape),
                strict=True,
            )
        )
        for tensor in tensors
    )


def slice_scatter_chunking_pass(graph: torch.fx.Graph) -> bool:
    """Rewrite complete slice_scatter chains without full-sized intermediates."""
    slice_scatters = graph.find_nodes(
        op="call_function", target=aten.slice_scatter.default
    )
    if len(slice_scatters) < 2:
        return False

    compute_mutation_region_ids(graph)
    introduced_mutation = False
    node_order = {node: index for index, node in enumerate(graph.nodes)}
    examined = OrderedSet[torch.fx.Node]()
    for slice_scatter in reversed(slice_scatters):
        # Only consider maximal chains. Retrying every rejected prefix makes a
        # long unsupported chain quadratic and is not needed by the target use case.
        if slice_scatter._erased or slice_scatter in examined:
            continue
        match = _SLICE_SCATTER_PATTERN.match(slice_scatter)
        # A full-tensor chain overwrites the base through adjacent slices with
        # no gaps or overlaps, so the original base values are unobservable.
        if not isinstance(match, Match):
            continue

        result = _get_full_tensor_slice_scatter_chain(slice_scatter, examined)
        if result is None:
            continue

        match.nodes = sorted(result.removable_nodes, key=node_order.__getitem__)

        if not _is_reusable_base_factory(result.base):
            continue
        if _custom_context(result.base) != _custom_context(result.slice_scatters[-1]):
            continue

        if _can_fuse_as_cat(result):

            def replacement(*chunks):
                return torch.cat(chunks, dim=result.dim)

            match.replace_by_example(replacement, result.sources)
            if not result.base.users and not result.base.is_impure():
                graph.erase_node(result.base)
            continue

        _replace_with_inplace_copies(match, result)
        introduced_mutation = True
    if introduced_mutation:
        # Each mutated base is writable storage disjoint from its inputs and has
        # no users outside its chain, so rewrites cannot invalidate later matches.
        compute_mutation_region_ids(graph)
    return introduced_mutation
