"""Reassociate chains of matmuls into the parenthesization with the fewest FLOPs.

Matmul is associative, but its cost is not: for `A @ B @ C` evaluating left to
right, as written, is only cheapest when the intermediate `A @ B` is not wider
than the operands. Attention output projection (`attn @ V @ W_o`) hits the other
case whenever `W_o` narrows the output, where folding `V @ W_o` first is cheaper
by the ratio of the two inner dimensions.
"""

import logging

import torch
from torch._dynamo.utils import counters


log = logging.getLogger(__name__)
aten = torch.ops.aten

_MATMUL_OPS = (aten.mm.default, aten.bmm.default)


def _static_shape(node: object) -> tuple[int, ...] | None:
    if not isinstance(node, torch.fx.Node):
        return None
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return None
    shape = tuple(val.shape)
    if not all(isinstance(dim, int) for dim in shape):
        return None
    return shape  # type: ignore[return-value]


def _is_matmul(node: object, target: object) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is target
    )


def _collect_chain(
    node: torch.fx.Node,
    target: object,
    leaves: list[torch.fx.Node],
    interior: list[torch.fx.Node],
) -> int | None:
    """Flatten the matmul chain rooted at `node`, returning its cost in MACs.

    `leaves` is filled left to right with the matrices being multiplied and
    `interior` in preorder with the matmul nodes being replaced. Returns None if
    the chain has a shape we cannot reason about statically.
    """
    interior.append(node)
    cost = 0
    for arg in node.args:
        if _is_matmul(arg, target) and len(arg.users) == 1:
            sub_cost = _collect_chain(arg, target, leaves, interior)  # type: ignore[arg-type]
            if sub_cost is None:
                return None
            cost += sub_cost
        elif _static_shape(arg) is not None:
            leaves.append(arg)  # type: ignore[arg-type]
        else:
            return None

    lhs_shape, rhs_shape = (_static_shape(arg) for arg in node.args)
    if lhs_shape is None or rhs_shape is None:
        return None
    return cost + lhs_shape[-2] * lhs_shape[-1] * rhs_shape[-1]


def _optimal_order(dims: list[int]) -> tuple[int, list[list[int]]]:
    """Classic matrix chain order DP over the shared dimensions `dims`."""
    n = len(dims) - 1
    cost = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            best = None
            for k in range(i, j):
                macs = dims[i] * dims[k + 1] * dims[j + 1]
                candidate = cost[i][k] + cost[k + 1][j] + macs
                if best is None or candidate < best:
                    best = candidate
                    split[i][j] = k
            cost[i][j] = best  # type: ignore[assignment]
    return cost[0][n - 1], split


def _rebuild(
    graph: torch.fx.Graph,
    target: object,
    leaves: list[torch.fx.Node],
    split: list[list[int]],
    i: int,
    j: int,
) -> torch.fx.Node:
    if i == j:
        return leaves[i]
    k = split[i][j]
    lhs = _rebuild(graph, target, leaves, split, i, k)
    rhs = _rebuild(graph, target, leaves, split, k + 1, j)
    node = graph.call_function(target, (lhs, rhs))  # type: ignore[arg-type]
    node.meta["val"] = target(lhs.meta["val"], rhs.meta["val"])  # type: ignore[operator]
    return node


def _chain_root(node: torch.fx.Node) -> bool:
    if len(node.users) != 1:
        return True
    (user,) = node.users
    return not (_is_matmul(user, node.target) and node in user.args)


def reassociate_matmul(graph: torch.fx.Graph) -> None:
    handled: set[torch.fx.Node] = set()
    for node in list(graph.nodes):
        if (
            node in handled
            or node.op != "call_function"
            or node.target not in _MATMUL_OPS
            or not _chain_root(node)
        ):
            continue

        leaves: list[torch.fx.Node] = []
        interior: list[torch.fx.Node] = []
        current_cost = _collect_chain(node, node.target, leaves, interior)
        handled.update(interior)
        if current_cost is None or len(leaves) < 3:
            continue

        shapes = [_static_shape(leaf) for leaf in leaves]
        batch = shapes[0][:-2]  # type: ignore[index]
        if any(shape[:-2] != batch for shape in shapes):  # type: ignore[index]
            continue

        dims = [shapes[0][-2]] + [shape[-1] for shape in shapes]  # type: ignore[index]
        best_cost, split = _optimal_order(dims)
        if best_cost >= current_cost:
            continue

        with graph.inserting_before(node):
            new_node = _rebuild(graph, node.target, leaves, split, 0, len(leaves) - 1)
        new_node.meta.update(
            {k: v for k, v in node.meta.items() if k not in ("val", "tensor_meta")}
        )
        node.replace_all_uses_with(new_node)
        for dead in interior:
            graph.erase_node(dead)

        counters["inductor"]["reassociate_matmul"] += 1
        log.debug(
            "reassociated %d-matmul chain, %d -> %d MACs",
            len(leaves),
            current_cost,
            best_cost,
        )
