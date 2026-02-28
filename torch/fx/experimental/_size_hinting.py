"""
Size hinting utilities for symbolic shape expressions.

This module contains the core logic for resolving symbolic expressions to
concrete integer hints. Two strategies are provided:

- _guarding_hint_or_throw_base: strict, only uses backed symbol hints, throws on
  unbacked symbols. Use for correctness-critical guarding decisions.
- _optimization_hint_base: permissive, uses heuristics and fallbacks for unbacked
  symbols. Use for performance optimization decisions.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional, TYPE_CHECKING, Union

import sympy

from torch.utils._sympy.numbers import int_oo


log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv


def _sympy_subs(expr: sympy.Expr, replacements: dict[sympy.Expr, Any]) -> sympy.Expr:
    """
    When the passed replacement symbol v is a string, it is converted to a symbol with name v that
    have the same replaced expression integer and nonnegative properties.
    """

    def to_symbol(
        replaced: sympy.Expr, replacement: Union[sympy.Expr, str]
    ) -> sympy.Symbol:
        if not isinstance(replaced, sympy.Expr):
            raise AssertionError(
                f"Expected sympy.Expr key, got {type(replaced)}: {replaced}"
            )
        if isinstance(replacement, str):
            return sympy.Symbol(
                replacement,
                integer=replaced.is_integer,  # type: ignore[attr-defined]
                nonnegative=replaced.is_nonnegative,  # type: ignore[attr-defined]
            )
        else:
            return replacement

    # xreplace is faster than subs, but is way more picky
    return sympy.sympify(expr).xreplace(
        {k: to_symbol(k, v) for k, v in replacements.items()}
    )


def _maybe_realize_expr(
    expr: sympy.Expr, nan_fallback: Optional[int]
) -> Optional[Union[int, bool]]:
    """
    Handle special sympy values in hinting APIs.

    Returns:
        - True/False for sympy.true/sympy.false (preserves bool type)
        - Raises ValueError for complex numbers
        - sys.maxsize for positive infinity
        - -sys.maxsize for negative infinity
        - fallback for NaN
        - None if no special handling needed
    """
    if expr is sympy.true:
        return True
    if expr is sympy.false:
        return False

    try:
        return int(expr)
    except (TypeError, ValueError):
        pass

    if isinstance(expr, sympy.Expr):
        if expr.has(sympy.I):
            raise ValueError(
                f"_maybe_realize_expr received a complex expression: {expr}. "
                "Tensor dimensions cannot be complex numbers."
            )
        if expr in (int_oo, sympy.oo):
            return sys.maxsize
        if expr in (-int_oo, -sympy.oo):
            return -sys.maxsize
        if nan_fallback is not None and (expr is sympy.nan or expr.has(sympy.nan)):
            return nan_fallback

    return None


def _guarding_hint_or_throw_base(
    shape_env: ShapeEnv,
    expr: Union[sympy.Expr, sympy.Basic, int, bool],
    precomputed_replacements: dict[sympy.Expr, sympy.Symbol],
) -> Union[int, bool]:
    """
    Return a concrete integer hint for an expression that is safe to use for guarding.

    This function evaluates the expression using only backed-symbols hints. Unlike
    _optimization_hint_base(), this function does NOT use heuristics or fallback values
    for unbacked symbols.

    Use this when you need a hint value that will be used for a guarding decision.

    Args:
        shape_env: The ShapeEnv instance.
        expr: A sympy expression or integer to evaluate.
        precomputed_replacements: Precomputed replacements for PRECOMPUTED_SIZE symbols.

    Returns:
        The concrete integer value of the expression based on backed symbol hints.

    Raises:
        GuardOnDataDependentSymNode: If the expression contains unbacked symbols
        (data-dependent values) that cannot be resolved to concrete values.

    See Also:
        _optimization_hint_base: For cases where fallback/heuristic values are acceptable
            for unbacked symbols.
    """
    from torch.fx.experimental.symbolic_shapes import (
        has_free_unbacked_symbols,
        symbol_is_type,
        SymT,
    )

    expr = sympy.expand(expr).xreplace(shape_env.replacements)

    if isinstance(expr, sympy.Expr):
        expr = expr.expand(identity=True)

    result = _maybe_realize_expr(expr, None)
    if result is not None:
        return result

    if not isinstance(expr, sympy.Basic):
        raise RuntimeError("isinstance(expr, sympy.Basic)", expr, type(expr))

    if any(symbol_is_type(s, SymT.PRECOMPUTED_SIZE) for s in expr.free_symbols):  # type: ignore[attr-defined]
        expr = _sympy_subs(expr, precomputed_replacements)  # type: ignore[arg-type]

    # TODO do we need sympy_subs, or just xreplace
    expr = _sympy_subs(expr, shape_env.backed_var_to_val)
    expr = expr.expand(identity=True)

    if has_free_unbacked_symbols(expr):
        # Note: we could do better here and call
        # _maybe_evaluate_static(orig_expr, compute_hint=True)
        # but is it worth the overhead? probably not.
        raise shape_env._make_data_dependent_error(expr, expr)

    result = _maybe_realize_expr(expr, None)
    if result is None:
        raise RuntimeError("unexpected None!", expr)
    return result


def _get_unbacked_replacements(shape_env: ShapeEnv) -> dict[sympy.Expr, sympy.Expr]:
    """Builds a mapping from unbacked expressions to canonical equivalents
    using a union-find algorithm over deferred runtime asserts.
    Used by optimization_hint to resolve unbacked symbols to consistent values."""
    from collections import defaultdict

    from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols
    from torch.utils._ordered_set import OrderedSet

    if shape_env._unbacked_replacements is not None:
        return shape_env._unbacked_replacements

    class CanonicalExprFinder:
        """
        A disjoint-set/union-find data structure that can return the
        "canonical" expression for a group of equivalent expressions.
        - The canonical expression must come from the input eq_graph.
        - The heuristics used to choose a leader determines which
        expression becomes the canonical expression.
        """

        def __init__(self, eq_graph: dict[sympy.Expr, OrderedSet[sympy.Expr]]):
            self.eq_graph = eq_graph
            self.expressions = list(eq_graph.keys())
            self.reverse_expressions = {
                expr: i for i, expr in enumerate(self.expressions)
            }
            self.leader = list(range(len(self.expressions)))
            self.size = [1] * len(self.expressions)
            self._build_canonical_expr_mapping()

        def _build_canonical_expr_mapping(self):
            for expr, edges in self.eq_graph.items():
                for adj in edges:
                    self.union_expr(expr, adj)

        def union_expr(self, a: sympy.Expr, b: sympy.Expr):
            return self.union(self.reverse_expressions[a], self.reverse_expressions[b])

        def union(self, a: int, b: int):
            rootA = self.find(a)
            rootB = self.find(b)
            if rootA == rootB:
                return False
            leader, other = self.choose_leader(rootA, rootB)
            self.leader[other] = leader
            self.size[leader] += self.size[other]
            return True

        def find_expr(self, expr: sympy.Expr):
            parent = self.find(self.reverse_expressions[expr])
            return self.expressions[parent]

        def find(self, x: int):
            if self.leader[x] != x:
                self.leader[x] = self.find(self.leader[x])
            return self.leader[x]

        def choose_leader(self, a: int, b: int):
            """
            The leader will become the canonical expression.
            Returns a (leader, follower) tuple.

            Heuristics:
            1. Backed expression or constants preferred over unbacked expr
            2. Simpler sub-expr when one contains the other
            3. Higher frequency across equalities from deferred runtime assertions
            4. Size of the set
            5. Fallback to sympy.Basic.compare
            """

            def _choose(x: int, y: int) -> bool:
                lhs, rhs = self.expressions[x], self.expressions[y]

                any_unbacked_lhs = has_free_unbacked_symbols(lhs)
                any_unbacked_rhs = has_free_unbacked_symbols(rhs)
                if any_unbacked_lhs != any_unbacked_rhs:
                    return bool(any_unbacked_rhs)

                if lhs.has(rhs):
                    return False
                elif rhs.has(lhs):
                    return True

                degrees_lhs = len(self.eq_graph[lhs])
                degrees_rhs = len(self.eq_graph[rhs])
                if degrees_lhs != degrees_rhs:
                    return degrees_lhs > degrees_rhs

                if self.size[x] != self.size[y]:
                    return self.size[x] > self.size[y]

                return lhs.compare(rhs) == -1

            if _choose(a, b):
                return a, b
            return b, a

    # Build an undirected graph using ShapeEnv's deferred runtime assertions.
    shape_env._equality_graph = defaultdict(OrderedSet)
    for assertions in shape_env.deferred_runtime_asserts.values():
        for assertion in assertions:
            if not isinstance(assertion.expr, sympy.Equality):
                continue
            lhs = sympy.sympify(assertion.expr.lhs)
            rhs = sympy.sympify(assertion.expr.rhs)
            shape_env._equality_graph[lhs].add(rhs)
            shape_env._equality_graph[rhs].add(lhs)

    uf = CanonicalExprFinder(shape_env._equality_graph)

    shape_env._unbacked_replacements = {}
    for expr in shape_env._equality_graph:
        canonical_expr = uf.find_expr(expr)
        if expr != canonical_expr:
            shape_env._unbacked_replacements[expr] = canonical_expr

    return shape_env._unbacked_replacements


def _sub_unbacked_exprs(shape_env: ShapeEnv, expr: sympy.Expr) -> sympy.Expr:
    """Substitute unbacked expressions with canonical equivalents.
    Used by optimization_hint to maximize consistency when hinting unbacked symbols."""
    replacements = _get_unbacked_replacements(shape_env)

    # consider making this threshold configurable
    sub_cnt_limit = 30
    sub_cnt = 0
    while sub_cnt < sub_cnt_limit:
        new_expr = expr.subs(replacements)
        if new_expr == expr:
            break
        expr = sympy.factor(new_expr)
        sub_cnt += 1
    else:
        log.warning("Substitution limit (%d) reached w/ %s", sub_cnt_limit, expr)

    expr = _sympy_subs(expr, shape_env.backed_var_to_val)
    expr = _sympy_subs(expr, shape_env.var_to_hint_override)
    return expr


def _optimization_hint_base(
    shape_env: ShapeEnv,
    expr: Union[sympy.Expr, int],
    precomputed_replacements: dict[sympy.Expr, sympy.Symbol],
    fallback: Optional[int] = None,
) -> int:
    """
    Return a concrete integer hint for an expression using heuristics.

    This function should be used for non-guarding based optimizations.
    It will hint unbacked symbols using user provided optimization hints.
    If not provided, fallback will be used along with some heuristics
    that try to maximize consistency with the shape environment.

    Args:
        shape_env: The ShapeEnv instance.
        expr: A sympy expression or integer to evaluate.
        precomputed_replacements: Precomputed replacements for PRECOMPUTED_SIZE symbols.
        fallback: Fallback value for unbacked symbols. If None, reads from config.

    Returns:
        A concrete integer hint for the expression.
    """
    from torch.fx.experimental.symbolic_shapes import (
        has_free_unbacked_symbols,
        symbol_is_type,
        SymT,
    )

    # Read config at call time to respect runtime patches (e.g., in tests)
    if fallback is None:
        from torch._inductor.config import unbacked_symint_fallback

        fallback = unbacked_symint_fallback

    original = expr
    expr = sympy.expand(expr).xreplace(shape_env.replacements)

    result = _maybe_realize_expr(expr, fallback)
    if result is not None:
        return result

    if isinstance(expr, sympy.Expr):
        expr = expr.expand(identity=True)

    # Replace backed symbols with their hints, leaving unbacked symbols alone.
    result = _maybe_realize_expr(expr, None)
    if result is not None:
        return result

    if not isinstance(expr, sympy.Expr):
        raise RuntimeError("isinstance(expr, sympy.Expr)", expr)

    if any(symbol_is_type(s, SymT.PRECOMPUTED_SIZE) for s in expr.free_symbols):  # type: ignore[attr-defined]
        expr = _sympy_subs(expr, precomputed_replacements)  # type: ignore[arg-type]

    expr = _sympy_subs(expr, shape_env.backed_var_to_val)
    expr = expr.expand(identity=True)

    result = _maybe_realize_expr(expr, fallback)
    if result is not None:
        return result

    expr = _sympy_subs(expr, shape_env.var_to_hint_override)

    result = _maybe_realize_expr(expr, fallback)
    if result is not None:
        return result

    # If unbacked symbols remain, try to substitute them using heuristics
    # that maximize consistency with the shape environment.
    if has_free_unbacked_symbols(expr):
        # Make sure to substitute with the factored version
        # e.g. 10*(s0 + u0) instead of 10*s0 + 10*u0
        # TODO optimize _sub_unbacked_exprs
        expr = _sub_unbacked_exprs(shape_env, sympy.factor(original))

    # For multiple expressions that depend on an unbacked symint,
    # we want to compute them consistently for a size hint we have chosen.
    # So, recursively compute expressions via size hints of contained symbols.
    # For example: u1 * u2 - 10 ==> fallback * fallback - 10

    if not isinstance(expr, sympy.Expr):
        raise RuntimeError(f"Expected sympy Expr, got {type(expr)}: {expr}")
    free_symbols = expr.free_symbols

    # Constrain fallback per-symbol based on var_to_range bounds
    size_dict = {}
    for s in free_symbols:
        sym_fallback = fallback
        vr = shape_env.var_to_range.get(s, None)
        if vr is not None:
            if isinstance(vr.lower, (int, sympy.Integer)):
                sym_fallback = max(sym_fallback, int(vr.lower))
            if isinstance(vr.upper, (int, sympy.Integer)):
                sym_fallback = min(sym_fallback, int(vr.upper))
        size_dict[s] = sym_fallback

    final_result = expr.subs(size_dict)

    final_result = _maybe_realize_expr(final_result, fallback)
    if final_result is None:
        raise RuntimeError(f"Failed to realize expression to int: {expr}")

    return final_result
