# mypy: allow-untyped-defs

import dataclasses

import sympy

from torch.utils._ordered_set import OrderedSet

from ...virtualized import V


@dataclasses.dataclass(frozen=True)
class LaneExprInfo:
    """Lane-wise uniformity and contiguity classification for an expression."""

    is_uniform: bool
    is_contiguous: bool
    contiguous_width: int | None


def classify_lane_expr(
    expr: sympy.Expr,
    lane_var: sympy.Symbol,
    *,
    max_width: int,
    uniform_symbols: OrderedSet[sympy.Symbol] | None = None,
) -> LaneExprInfo:
    """Classify whether an expression is uniform or contiguous across lanes."""
    if max_width <= 1:
        return LaneExprInfo(True, False, None)

    semantic_expr = V.graph.sizevars.simplify(expr)
    if uniform_symbols is not None and semantic_expr.free_symbols <= uniform_symbols:
        return LaneExprInfo(True, False, None)

    lane_contiguity = V.graph.sizevars.analyze_lane_contiguity(semantic_expr, lane_var)
    contiguous_width = aligned_contiguous_width(
        semantic_expr,
        lane_var,
        max_width=max_width,
        lane_contiguity=lane_contiguity,
    )
    return LaneExprInfo(
        is_uniform=lane_contiguity.is_uniform_for(max_width),
        is_contiguous=lane_contiguity.stride == 1 and contiguous_width == max_width,
        contiguous_width=contiguous_width,
    )


def aligned_contiguous_width(
    expr: sympy.Expr,
    lane_var: sympy.Symbol,
    *,
    max_width: int,
    lane_contiguity=None,
) -> int | None:
    """Return the largest aligned power-of-two contiguous lane width."""
    assert max_width >= 2 and max_width.bit_count() == 1
    if lane_contiguity is None:
        lane_contiguity = V.graph.sizevars.analyze_lane_contiguity(expr, lane_var)
    start_expr = lane_group_start(expr, lane_var)
    width = max_width
    while width >= 2:
        if (
            lane_contiguity.is_contiguous_for(width)
            and V.graph.sizevars.statically_known_multiple_of(start_expr, width)
            and V.graph.sizevars.statically_known_geq(start_expr, 0)
        ):
            return width
        width //= 2
    return None


def lane_group_start(expr: sympy.Expr, lane_var: sympy.Symbol) -> sympy.Expr:
    """Return the expression value at the first lane in the lane group."""
    return V.graph.sizevars.simplify(expr.xreplace({lane_var: sympy.Integer(0)}))


def decompose_affine_lane_expr(
    expr: sympy.Expr, lane_var: sympy.Symbol
) -> tuple[sympy.Expr, sympy.Expr] | None:
    """Decompose an affine lane expression into ``(lane_coeff, rest)``."""
    expr = V.graph.sizevars.simplify(expr)
    lane_coeff = expr.coeff(lane_var)
    rest = lane_group_start(expr, lane_var)
    if lane_var in lane_coeff.free_symbols or lane_var in rest.free_symbols:
        return None
    residual = V.graph.sizevars.simplify(expr - (lane_coeff * lane_var + rest))
    if residual != 0:
        return None
    return lane_coeff, rest
