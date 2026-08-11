import math
import operator
from typing import Any

import torch
from torch.fx import Graph, map_arg, Node
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_flatten


# flattens with support for slices
# Note: a better way to do this would
# be register/unregister slices as pytree nodes
# but there is no unregister API in the pytorch
# pytree impl
def _get_flat_args(
    node: Node, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> list[Node]:
    args = list[Any]()
    map_arg((node.args, node.kwargs), args.append)
    if node in node_to_additional_deps:
        args.extend(node_to_additional_deps[node])
    return args


def _get_flat_args_unique(
    node: Node, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> OrderedSet[Node]:
    args = OrderedSet[Node]()
    map_arg((node.args, node.kwargs), args.add)
    if node in node_to_additional_deps:
        args.update(node_to_additional_deps[node])
    return args


def _detect_cycles(
    graph: Graph, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> str:
    # States: 0=Unvisited, 1=Visiting, 2=Visited(Safe)
    state: dict[Node, int] = {}

    for root in reversed(graph.nodes):
        if root in state:
            continue

        # Stack holds (current_node, children_iterator).
        # Using an iterator allows us to pause and resume processing a node's children.
        stack = [(root, iter(_get_flat_args_unique(root, node_to_additional_deps)))]
        state[root] = 1  # Visiting

        while stack:
            parent, children = stack[-1]

            try:
                child = next(children)

                if not isinstance(child, Node):
                    continue

                child_state = state.get(child, 0)

                if child_state == 1:
                    # Back-edge: child is on the current DFS path -> cycle
                    cycle_path = [node for node, _ in stack] + [child]
                    return f"cycle detected in path: {cycle_path}"

                if child_state == 0:
                    state[child] = 1
                    stack.append(
                        (
                            child,
                            iter(_get_flat_args_unique(child, node_to_additional_deps)),
                        )
                    )
                # child_state == 2 means already verified safe; skip.

            except StopIteration:
                # All children processed — mark safe and pop.
                stack.pop()
                state[parent] = 2

    return "no cycle detected"


def _graph_device_type(graph: Graph | None) -> str:
    if graph is None:
        return "cpu"

    def _device_type(x: Any) -> str:
        if isinstance(x, torch.device):
            return x.type
        if isinstance(x, torch.Tensor):
            return x.device.type
        return "cpu"

    def _flatten_meta(node: Node, key: str) -> list[Any]:
        if key not in node.meta:
            return []
        flat, _ = tree_flatten(node.meta[key])
        return flat

    for node in graph.nodes:
        for key in ("val", "example_value"):
            for obj in _flatten_meta(node, key):
                return _device_type(obj)

        # Check for device conversions
        if node.op == "call_method":
            for gpu in ["cuda", "xpu"]:
                if node.target == gpu:
                    return gpu
                if node.target == "to" and gpu in node.args:
                    return gpu

        # Check args/kwargs for non-CPU device specs
        flat_args, _ = tree_flatten((node.args, node.kwargs))
        for obj in flat_args:
            return _device_type(obj)
    return "cpu"


_SDPA_TARGETS = {
    torch._C._nn.scaled_dot_product_attention,
    torch.nn.functional.scaled_dot_product_attention,
    torch.ops.aten.scaled_dot_product_attention.default,
}

_SDPA_QUERY_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}

_GE_TARGETS = {operator.ge}

_ADD_TARGETS = {operator.add}

_SUB_TARGETS = {operator.sub}

_ARANGE_TARGETS = {torch.arange}

_ONES_TARGETS = {torch.ones}

_SCALAR_TARGETS = {torch.tensor}


def _node_meta_tensor(node: Node) -> torch.Tensor | None:
    for key in ("example_value", "val"):
        value = node.meta.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _node_dtype(node: Node) -> torch.dtype | None:
    value = _node_meta_tensor(node)
    return value.dtype if value is not None else None


def _is_exact_number(value: object, expected: int | float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return value == expected


def _is_finite_scalar_in_dtype(value: object, dtype: torch.dtype) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    dtype_max = torch.finfo(dtype).max
    return -dtype_max <= value <= dtype_max


def _get_sdpa_mask_node(node: Node) -> Node | None:
    if "attn_mask" in node.kwargs:
        mask = node.kwargs["attn_mask"]
    elif len(node.args) >= 4:
        mask = node.args[3]
    else:
        mask = None
    return mask if isinstance(mask, Node) else None


def _get_sdpa_is_causal(node: Node) -> object:
    if "is_causal" in node.kwargs:
        return node.kwargs["is_causal"]
    if len(node.args) >= 6:
        return node.args[5]
    return False


def _sdpa_metadata_is_valid(node: Node, mask: Node) -> bool:
    query = node.args[0] if node.args else node.kwargs.get("query")
    if not isinstance(query, Node) or _node_meta_tensor(node) is None:
        return False

    query_meta = _node_meta_tensor(query)
    mask_meta = _node_meta_tensor(mask)
    if query_meta is None or mask_meta is None:
        return False
    if query_meta.layout is not torch.strided or mask_meta.layout is not torch.strided:
        return False
    if query_meta.device != mask_meta.device:
        return False
    if query_meta.dtype not in _SDPA_QUERY_DTYPES:
        return False
    return mask_meta.dtype is torch.bool or mask_meta.dtype is query_meta.dtype


def _replace_sdpa_mask_with_none(node: Node) -> None:
    if "attn_mask" in node.kwargs:
        kwargs = dict(node.kwargs)
        kwargs["attn_mask"] = None
        node.kwargs = kwargs
    else:
        args = list(node.args)
        args[3] = None
        node.args = tuple(args)


def _is_full_slice_index(index: object) -> bool:
    if not isinstance(index, tuple):
        return False
    for item in index:
        if item is None or item is Ellipsis:
            continue
        if isinstance(item, slice) and (
            item.start is None and item.stop is None and item.step is None
        ):
            continue
        return False
    return True


def _strip_shape_only_ops(node: Node, matched: set[Node]) -> Node:
    while True:
        if node.op == "call_method" and node.target == "expand" and node.args:
            source = node.args[0]
        elif (
            node.op == "call_function"
            and node.target is operator.getitem
            and len(node.args) >= 2
            and _is_full_slice_index(node.args[1])
        ):
            source = node.args[0]
        else:
            return node

        if not isinstance(source, Node):
            return node
        matched.add(node)
        node = source


def _arange_start_end_step(node: Node) -> tuple[object, object, object] | None:
    if node.target not in _ARANGE_TARGETS:
        return None

    if node.target is torch.arange:
        if "end" in node.kwargs:
            start = node.args[0] if node.args else node.kwargs.get("start", 0)
            end = node.kwargs["end"]
            step = node.args[1] if len(node.args) > 1 else node.kwargs.get("step", 1)
            return start, end, step
        if len(node.args) == 1:
            return 0, node.args[0], node.kwargs.get("step", 1)
        if len(node.args) >= 2:
            step = node.args[2] if len(node.args) > 2 else node.kwargs.get("step", 1)
            return node.args[0], node.args[1], step
        return None

    return None


def _match_default_arange(node: Node, matched: set[Node]) -> bool:
    values = _arange_start_end_step(node)
    if values is None or _node_dtype(node) is not torch.int64:
        return False
    start, _, step = values
    if not (_is_exact_number(start, 0) and _is_exact_number(step, 1)):
        return False
    matched.add(node)
    return True


def _strip_identity_add(node: Node, matched: set[Node]) -> Node:
    if node.op != "call_function" or node.target not in _ADD_TARGETS:
        return node
    if len(node.args) < 2 or not _is_exact_number(node.kwargs.get("alpha", 1), 1):
        return node

    lhs, rhs = node.args[:2]
    if isinstance(lhs, Node) and _is_exact_number(rhs, 0):
        matched.add(node)
        return lhs
    if isinstance(rhs, Node) and _is_exact_number(lhs, 0):
        matched.add(node)
        return rhs
    return node


def _match_bidirectional_boolean_mask(mask: Node) -> set[Node] | None:
    if _node_dtype(mask) is not torch.bool:
        return None

    matched: set[Node] = set()
    comparison = _strip_shape_only_ops(mask, matched)
    if (
        comparison.op != "call_function"
        or comparison.target not in _GE_TARGETS
        or len(comparison.args) < 2
        or not _is_exact_number(comparison.args[1], 0)
    ):
        return None
    matched.add(comparison)

    indices = comparison.args[0]
    if not isinstance(indices, Node):
        return None
    indices = _strip_shape_only_ops(indices, matched)
    indices = _strip_identity_add(indices, matched)
    if not _match_default_arange(indices, matched):
        return None
    return matched


def _requested_to_dtype(node: Node) -> torch.dtype | None:
    dtype = node.kwargs.get("dtype")
    if isinstance(dtype, torch.dtype):
        return dtype
    for arg in node.args[1:]:
        if isinstance(arg, torch.dtype):
            return arg
    return None


def _match_to_dtype(node: Node, dtype: torch.dtype, matched: set[Node]) -> Node | None:
    if (
        node.op != "call_method"
        or node.target != "to"
        or not node.args
        or _requested_to_dtype(node) is not dtype
        or _node_dtype(node) is not dtype
    ):
        return None
    source = node.args[0]
    if not isinstance(source, Node):
        return None
    matched.add(node)
    return source


def _match_scalar_one(node: object, dtype: torch.dtype, matched: set[Node]) -> bool:
    if (
        not isinstance(node, Node)
        or node.op != "call_function"
        or node.target not in _SCALAR_TARGETS
        or not node.args
        or not _is_exact_number(node.args[0], 1)
        or _node_dtype(node) is not dtype
    ):
        return False
    matched.add(node)
    return True


def _match_expanded_ones(node: Node, dtype: torch.dtype, matched: set[Node]) -> bool:
    if node.op == "call_method" and node.target == "to":
        source = _match_to_dtype(node, dtype, matched)
        if source is None:
            return False
        node = source
    elif _node_dtype(node) is not dtype:
        return False

    node = _strip_shape_only_ops(node, matched)
    node_meta = _node_meta_tensor(node)
    if (
        node.op != "call_function"
        or node.target not in _ONES_TARGETS
        or node_meta is None
        or not node_meta.dtype.is_floating_point
    ):
        return False
    matched.add(node)
    return True


def _match_transformers_additive_mask(mask: Node) -> set[Node] | None:
    mask_meta = _node_meta_tensor(mask)
    if (
        mask_meta is None
        or not mask_meta.dtype.is_floating_point
        or mask_meta.requires_grad
        or mask.op != "call_method"
        or mask.target != "masked_fill"
        or len(mask.args) < 3
    ):
        return None

    inverted, condition, fill_value = mask.args[:3]
    if not isinstance(inverted, Node) or not isinstance(condition, Node):
        return None
    if not _is_finite_scalar_in_dtype(fill_value, mask_meta.dtype):
        return None

    matched = {mask}
    condition_source = _match_to_dtype(condition, torch.bool, matched)
    if condition_source is not inverted:
        return None

    if (
        inverted.op != "call_function"
        or inverted.target not in _SUB_TARGETS
        or len(inverted.args) < 2
        or not _is_exact_number(inverted.kwargs.get("alpha", 1), 1)
        or _node_dtype(inverted) is not mask_meta.dtype
    ):
        return None
    matched.add(inverted)

    one, expanded_ones = inverted.args[:2]
    if not _match_scalar_one(one, mask_meta.dtype, matched):
        return None
    if not isinstance(expanded_ones, Node) or not _match_expanded_ones(
        expanded_ones, mask_meta.dtype, matched
    ):
        return None
    return matched


def _matched_nodes_are_safe(matched: set[Node]) -> bool:
    for node in matched:
        value = _node_meta_tensor(node)
        if value is None or value.requires_grad or node.is_impure():
            return False
        if "out" in node.kwargs and node.kwargs["out"] is not None:
            return False
        for user in node.users:
            if user in matched:
                continue
            if user.op == "call_function" and user.target in _SDPA_TARGETS:
                continue
            return False
    return True


def remove_noop_sdpa_masks(gm: torch.fx.GraphModule) -> bool:
    """
    Remove the two structurally no-op SDPA masks emitted by Transformers.

    Bidirectional attention can materialize ``arange(...) >= 0`` as an all-true
    boolean mask. Its additive-mask helper materializes ``1 - ones`` followed by
    ``masked_fill`` as an all-zero mask. Both prevent CUDA flash attention
    dispatch. This pass deliberately matches only those exact producer graphs,
    after successful fake execution, and only when their producers are pure and
    used exclusively by the matched mask and SDPA calls.
    """
    changed = False
    dead_candidates: set[Node] = set()
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in _SDPA_TARGETS:
            continue
        mask = _get_sdpa_mask_node(node)
        if (
            mask is None
            or _get_sdpa_is_causal(node) is not False
            or not _sdpa_metadata_is_valid(node, mask)
        ):
            continue

        matched = _match_bidirectional_boolean_mask(mask)
        if matched is None:
            matched = _match_transformers_additive_mask(mask)
        if matched is None or not _matched_nodes_are_safe(matched):
            continue

        _replace_sdpa_mask_with_none(node)
        dead_candidates.update(matched)
        changed = True

    if changed:
        for node in reversed(gm.graph.nodes):
            if node in dead_candidates and not node.users:
                gm.graph.erase_node(node)
        gm.graph.lint()
        gm.recompile()

    return changed
