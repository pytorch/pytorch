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

_PRESERVE_VALUES_FUNCTION_TARGETS = {
    operator.getitem,
    torch.ops.aten.alias.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.select.int,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.transpose.int,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.view.default,
}

_PRESERVE_VALUES_METHOD_TARGETS = {
    "clone",
    "contiguous",
    "detach",
    "expand",
    "permute",
    "reshape",
    "squeeze",
    "transpose",
    "unsqueeze",
    "view",
}

_BITWISE_AND_TARGETS = {
    operator.and_,
    torch.bitwise_and,
    torch.logical_and,
    torch.ops.aten.bitwise_and.Tensor,
    torch.ops.aten.logical_and.default,
}

_GE_TARGETS = {
    operator.ge,
    torch.ge,
    torch.ops.aten.ge.Scalar,
    torch.ops.aten.ge.Tensor,
}

_ADD_TARGETS = {
    operator.add,
    torch.add,
    torch.ops.aten.add.Scalar,
    torch.ops.aten.add.Tensor,
}

_SUB_TARGETS = {
    operator.sub,
    torch.sub,
    torch.ops.aten.sub.Scalar,
    torch.ops.aten.sub.Tensor,
}

_MUL_TARGETS = {
    operator.mul,
    torch.mul,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.mul.Tensor,
}

_ARANGE_TARGETS = {
    torch.arange,
    torch.ops.aten.arange.default,
    torch.ops.aten.arange.start,
    torch.ops.aten.arange.start_step,
}

_ONES_TARGETS = {
    torch.ones,
    torch.ones_like,
    torch.ops.aten.ones.default,
    torch.ops.aten.ones_like.default,
}

_SCALAR_TARGETS = {
    torch.tensor,
    torch.ops.aten.scalar_tensor.default,
}

_ZEROS_TARGETS = {
    torch.zeros,
    torch.zeros_like,
    torch.ops.aten.zeros.default,
    torch.ops.aten.zeros_like.default,
}

_MASKED_FILL_TARGETS = {
    torch.masked_fill,
    torch.ops.aten.masked_fill.Scalar,
    torch.ops.aten.masked_fill.Tensor,
}


def _node_meta_tensor(node: Node) -> torch.Tensor | None:
    for key in ("example_value", "val"):
        value = node.meta.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _node_dtype(node: Node) -> torch.dtype | None:
    meta_tensor = _node_meta_tensor(node)
    if meta_tensor is not None:
        return meta_tensor.dtype
    return None


def _node_requires_grad(node: Node) -> bool:
    meta_tensor = _node_meta_tensor(node)
    if meta_tensor is not None:
        return meta_tensor.requires_grad
    return False


def _node_has_signed_integer_dtype(node: Node) -> bool:
    dtype = _node_dtype(node)
    if not isinstance(dtype, torch.dtype):
        return False

    try:
        info = torch.iinfo(dtype)
    except (TypeError, ValueError):
        return False

    return info.min < 0


def _is_nonnegative_constant(value: object) -> bool:
    return isinstance(value, (int, float)) and value >= 0


def _is_nonpositive_constant(value: object) -> bool:
    return isinstance(value, (int, float)) and value <= 0


def _constant_product_is_nonnegative(lhs: object, rhs: object) -> bool:
    return (
        isinstance(lhs, (int, float))
        and isinstance(rhs, (int, float))
        and lhs * rhs >= 0
    )


def _is_constant(value: object, constant: int | float | bool) -> bool:
    if isinstance(value, bool) or isinstance(constant, bool):
        return value is constant
    return isinstance(value, (int, float, bool)) and value == constant


def _fill_value(node: Node) -> object:
    if node.target is torch.full or node.target is torch.ops.aten.full.default:
        return node.args[1] if len(node.args) > 1 else node.kwargs.get("fill_value")
    if node.target in _SCALAR_TARGETS:
        return node.args[0] if node.args else None
    return None


def _first_arg(node: Node) -> object:
    if node.args:
        return node.args[0]
    return None


def _arange_start_end_step(node: Node) -> tuple[object, object, object] | None:
    if node.target not in _ARANGE_TARGETS or not node.args:
        return None

    if node.target in (torch.arange, torch.ops.aten.arange.default):
        if len(node.args) == 1:
            return 0, node.args[0], 1
        start = node.args[0]
        end = node.args[1]
        step = node.args[2] if len(node.args) > 2 else 1
        return start, end, step

    if node.target is torch.ops.aten.arange.start:
        return node.args[0], node.args[1], 1

    return node.args[0], node.args[1], node.args[2]


def _needs_explicit_arange_range_check(node: Node) -> bool:
    dtype = node.kwargs.get("dtype")
    if not isinstance(dtype, torch.dtype):
        return False

    try:
        info = torch.iinfo(dtype)
    except (TypeError, ValueError):
        return False

    return info.min < 0


def _explicit_arange_range_stays_nonnegative(node: Node) -> bool:
    values = _arange_start_end_step(node)
    if values is None:
        return False

    start, end, step = values
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or not isinstance(step, int)
        or isinstance(start, bool)
        or isinstance(end, bool)
        or isinstance(step, bool)
    ):
        return False

    if start < 0 or step <= 0:
        return False

    if end <= start:
        return True

    dtype = node.kwargs.get("dtype")
    if not isinstance(dtype, torch.dtype):
        return False
    dtype_max = torch.iinfo(dtype).max
    last_value = start + ((end - start - 1) // step) * step
    return last_value <= dtype_max


def _is_arange_start_nonnegative(node: Node) -> bool:
    values = _arange_start_end_step(node)
    if values is None:
        return False

    start, _, step = values

    if not (
        _is_nonnegative_constant(start) and isinstance(step, (int, float)) and step > 0
    ):
        return False

    if _needs_explicit_arange_range_check(node):
        return _explicit_arange_range_stays_nonnegative(node)

    return True


def _signed_integer_add_is_nonnegative(
    lhs: object, rhs: object, alpha: object, memo: dict[Node, bool]
) -> bool:
    if isinstance(lhs, Node) and (_is_constant(alpha, 0) or _is_constant(rhs, 0)):
        return _is_nonnegative_index_node(lhs, memo)

    if isinstance(rhs, Node) and _is_constant(lhs, 0) and _is_constant(alpha, 1):
        return _is_nonnegative_index_node(rhs, memo)

    return False


def _is_nonnegative_index_node(
    node: object, memo: dict[Node, bool] | None = None
) -> bool:
    if not isinstance(node, Node):
        return _is_nonnegative_constant(node)

    if memo is None:
        memo = {}
    if node in memo:
        return memo[node]
    memo[node] = False

    result = False
    if node.op == "call_function":
        if _is_arange_start_nonnegative(node):
            result = True
        elif node.target in _PRESERVE_VALUES_FUNCTION_TARGETS:
            result = _is_nonnegative_index_node(_first_arg(node), memo)
        elif node.target in _ADD_TARGETS and len(node.args) >= 2:
            lhs, rhs = node.args[0], node.args[1]
            alpha = node.kwargs.get("alpha", 1)
            if _node_has_signed_integer_dtype(node):
                result = _signed_integer_add_is_nonnegative(lhs, rhs, alpha, memo)
            else:
                result = (
                    _is_nonnegative_index_node(lhs, memo)
                    and _constant_product_is_nonnegative(alpha, rhs)
                ) or (
                    _is_nonnegative_constant(lhs)
                    and _is_nonnegative_constant(alpha)
                    and _is_nonnegative_index_node(rhs, memo)
                )
    elif node.op == "call_method":
        if node.target in _PRESERVE_VALUES_METHOD_TARGETS:
            result = _is_nonnegative_index_node(_first_arg(node), memo)

    memo[node] = result
    return result


def _is_known_one_additive_mask_node(
    node: object, memo: dict[Node, bool] | None = None
) -> bool:
    if not isinstance(node, Node):
        return _is_constant(node, 1)

    if memo is None:
        memo = {}
    if node in memo:
        return memo[node]
    memo[node] = False

    if _node_dtype(node) is torch.bool or _node_requires_grad(node):
        return False

    result = False
    if node.op == "call_function":
        if node.target in _ONES_TARGETS:
            result = True
        elif (
            node.target is torch.full
            or node.target is torch.ops.aten.full.default
            or node.target in _SCALAR_TARGETS
        ):
            result = _is_constant(_fill_value(node), 1)
        elif node.target in _PRESERVE_VALUES_FUNCTION_TARGETS:
            result = _is_known_one_additive_mask_node(_first_arg(node), memo)
    elif node.op == "call_method":
        if node.target == "to" or node.target in _PRESERVE_VALUES_METHOD_TARGETS:
            result = _is_known_one_additive_mask_node(_first_arg(node), memo)

    memo[node] = result
    return result


def _is_known_false_mask_node(
    node: object, memo: dict[Node, bool] | None = None
) -> bool:
    if not isinstance(node, Node):
        return node is False

    if memo is None:
        memo = {}
    if node in memo:
        return memo[node]
    memo[node] = False

    if _node_dtype(node) is not torch.bool:
        return False

    result = False
    if node.op == "call_function":
        if node.target in _ZEROS_TARGETS:
            result = True
        elif (
            node.target is torch.full
            or node.target is torch.ops.aten.full.default
            or node.target in _SCALAR_TARGETS
        ):
            result = _fill_value(node) is False
        elif node.target in _PRESERVE_VALUES_FUNCTION_TARGETS:
            result = _is_known_false_mask_node(_first_arg(node), memo)
        elif node.target in _BITWISE_AND_TARGETS and len(node.args) >= 2:
            result = any(_is_known_false_mask_node(arg, memo) for arg in node.args[:2])
    elif node.op == "call_method":
        if node.target == "to" and _is_known_zero_additive_mask_node(_first_arg(node)):
            result = True
        elif node.target == "to":
            result = _is_known_false_mask_node(_first_arg(node), memo)
        elif node.target in _PRESERVE_VALUES_METHOD_TARGETS:
            result = _is_known_false_mask_node(_first_arg(node), memo)

    memo[node] = result
    return result


def _is_finite_scalar_constant(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _is_zero_times_constant(node: Node, memo: dict[Node, bool]) -> bool:
    if len(node.args) < 2:
        return False
    lhs, rhs = node.args[0], node.args[1]
    return (
        _is_known_zero_additive_mask_node(lhs, memo) and _is_finite_scalar_constant(rhs)
    ) or (
        _is_finite_scalar_constant(lhs) and _is_known_zero_additive_mask_node(rhs, memo)
    )


def _is_known_zero_additive_mask_node(
    node: object, memo: dict[Node, bool] | None = None
) -> bool:
    if not isinstance(node, Node):
        return _is_constant(node, 0)

    if memo is None:
        memo = {}
    if node in memo:
        return memo[node]
    memo[node] = False

    if _node_dtype(node) is torch.bool or _node_requires_grad(node):
        return False

    result = False
    if node.op == "call_function":
        if node.target in _ZEROS_TARGETS:
            result = True
        elif (
            node.target is torch.full
            or node.target is torch.ops.aten.full.default
            or node.target in _SCALAR_TARGETS
        ):
            result = _is_constant(_fill_value(node), 0)
        elif node.target in _PRESERVE_VALUES_FUNCTION_TARGETS:
            result = _is_known_zero_additive_mask_node(_first_arg(node), memo)
        elif node.target in _ADD_TARGETS and len(node.args) >= 2:
            alpha = node.kwargs.get("alpha", 1)
            result = (
                _is_known_zero_additive_mask_node(node.args[0], memo)
                and _is_known_zero_additive_mask_node(node.args[1], memo)
                and _is_constant(alpha, 1)
            )
        elif node.target in _SUB_TARGETS and len(node.args) >= 2:
            alpha = node.kwargs.get("alpha", 1)
            result = _is_constant(alpha, 1) and (
                (
                    _is_known_zero_additive_mask_node(node.args[0], memo)
                    and _is_known_zero_additive_mask_node(node.args[1], memo)
                )
                or (
                    _is_known_one_additive_mask_node(node.args[0])
                    and _is_known_one_additive_mask_node(node.args[1])
                )
            )
        elif node.target in _MUL_TARGETS:
            result = _is_zero_times_constant(node, memo)
        elif node.target in _MASKED_FILL_TARGETS and len(node.args) >= 3:
            result = _is_known_zero_additive_mask_node(
                node.args[0], memo
            ) and _is_known_false_mask_node(node.args[1])
    elif node.op == "call_method":
        if node.target == "to" or node.target in _PRESERVE_VALUES_METHOD_TARGETS:
            result = _is_known_zero_additive_mask_node(_first_arg(node), memo)
        elif node.target == "masked_fill" and len(node.args) >= 3:
            result = _is_known_zero_additive_mask_node(
                node.args[0], memo
            ) and _is_known_false_mask_node(node.args[1])

    memo[node] = result
    return result


def _is_known_true_mask_node(
    node: object, memo: dict[Node, bool] | None = None
) -> bool:
    if not isinstance(node, Node):
        return node is True

    if memo is None:
        memo = {}
    if node in memo:
        return memo[node]
    memo[node] = False

    if _node_dtype(node) is not torch.bool:
        return False

    result = False
    if node.op == "call_function":
        if node.target in _ONES_TARGETS:
            result = True
        elif node.target is torch.full or node.target is torch.ops.aten.full.default:
            fill_value = (
                node.args[1] if len(node.args) > 1 else node.kwargs.get("fill_value")
            )
            result = fill_value is True
        elif node.target in _PRESERVE_VALUES_FUNCTION_TARGETS:
            result = _is_known_true_mask_node(_first_arg(node), memo)
        elif node.target in _BITWISE_AND_TARGETS and len(node.args) >= 2:
            result = all(_is_known_true_mask_node(arg, memo) for arg in node.args[:2])
        elif node.target in _GE_TARGETS and len(node.args) >= 2:
            result = _is_nonnegative_index_node(
                node.args[0]
            ) and _is_nonpositive_constant(node.args[1])
    elif node.op == "call_method":
        if node.target == "to" or node.target in _PRESERVE_VALUES_METHOD_TARGETS:
            result = _is_known_true_mask_node(_first_arg(node), memo)

    memo[node] = result
    return result


def _get_sdpa_mask_node(node: Node) -> Node | None:
    if "attn_mask" in node.kwargs:
        mask = node.kwargs["attn_mask"]
    elif len(node.args) >= 4:
        mask = node.args[3]
    else:
        mask = None
    return mask if isinstance(mask, Node) else None


def _replace_sdpa_mask_with_none(node: Node) -> None:
    if "attn_mask" in node.kwargs:
        kwargs = dict(node.kwargs)
        kwargs["attn_mask"] = None
        node.kwargs = kwargs
    else:
        args = list(node.args)
        args[3] = None
        node.args = tuple(args)


def remove_noop_sdpa_masks(gm: torch.fx.GraphModule) -> bool:
    """
    Remove SDPA masks that are provably no-ops from their FX producers.

    Some libraries avoid data-dependent ``mask.all()`` checks while tracing and
    materialize a full-attention boolean mask or zero additive mask instead. A
    non-null SDPA mask prevents CUDA flash attention dispatch even when the mask
    has no effect. This pass only handles masks proven no-op from graph
    structure, such as the bidirectional ``arange(...) >= 0`` mask used by
    Transformers.
    """
    changed = False
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in _SDPA_TARGETS:
            continue
        mask = _get_sdpa_mask_node(node)
        if mask is not None and (
            _is_known_true_mask_node(mask) or _is_known_zero_additive_mask_node(mask)
        ):
            _replace_sdpa_mask_with_none(node)
            changed = True

    if changed:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()

    return changed
