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
import math
import sys
from functools import lru_cache
from typing import Any, TYPE_CHECKING

import sympy

from torch.utils._sympy.numbers import int_oo


log = logging.getLogger(__name__)

# Maximum number of free symbols in an expression before we skip
# sympy.factor() in optimization_hint process for unbacked.
# Factoring polynomials with many variables is expensive.
SYMPY_FACTOR_MAX_FREE_SYMBOLS = 50

# Maximum product of expression free symbols and non-symbol replacement count
# before optimization_hint skips expensive SymPy subs() canonicalization.
SYMPY_SUBS_MAX_SYMBOL_REPLACEMENT_PRODUCT = 1024

# Maximum passes over the multi-symbol linear asserts when repairing unbacked hints.
CONSISTENT_HINTS_MAX_ITERS = 100

if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv


def _sympy_subs(expr: sympy.Basic, replacements: dict[sympy.Expr, Any]) -> sympy.Basic:
    """
    When the passed replacement symbol v is a string, it is converted to a symbol with name v that
    have the same replaced expression integer and nonnegative properties.
    """

    def to_symbol(replaced: sympy.Expr, replacement: sympy.Expr | str) -> sympy.Symbol:
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
    expr: sympy.Basic, nan_fallback: int | None
) -> int | bool | None:
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
    expr: sympy.Expr | sympy.Basic | int | bool,
    precomputed_replacements: dict[sympy.Expr, sympy.Symbol],
) -> int | bool:
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

    # sympy.expand() doesn't work with boolean expressions like Or/And
    if isinstance(expr, sympy.Expr):
        expr = sympy.expand(expr).xreplace(shape_env.replacements)
    else:
        expr = sympy.sympify(expr).xreplace(shape_env.replacements)

    if isinstance(expr, sympy.Expr):
        expr = expr.expand(identity=True)

    result = _maybe_realize_expr(expr, None)
    if result is not None:
        return result

    if not isinstance(expr, sympy.Basic):
        raise RuntimeError("isinstance(expr, sympy.Basic)", expr, type(expr))

    if any(symbol_is_type(s, SymT.PRECOMPUTED_SIZE) for s in expr.free_symbols):  # type: ignore[attr-defined]
        expr = _sympy_subs(expr, precomputed_replacements)

    # TODO do we need sympy_subs, or just xreplace
    expr = _sympy_subs(expr, shape_env.backed_var_to_val)
    if isinstance(expr, sympy.Expr):
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

        def __init__(self, eq_graph: dict[sympy.Expr, OrderedSet[sympy.Expr]]) -> None:
            self.eq_graph = eq_graph
            self.expressions = list(eq_graph.keys())
            self.reverse_expressions = {
                expr: i for i, expr in enumerate(self.expressions)
            }
            self.leader = list(range(len(self.expressions)))
            self.size = [1] * len(self.expressions)
            self._build_canonical_expr_mapping()

        def _build_canonical_expr_mapping(self) -> None:
            for expr, edges in self.eq_graph.items():
                for adj in edges:
                    self.union_expr(expr, adj)

        def union_expr(self, a: sympy.Expr, b: sympy.Expr) -> bool:
            return self.union(self.reverse_expressions[a], self.reverse_expressions[b])

        def union(self, a: int, b: int) -> bool:
            rootA = self.find(a)
            rootB = self.find(b)
            if rootA == rootB:
                return False
            leader, other = self.choose_leader(rootA, rootB)
            self.leader[other] = leader
            self.size[leader] += self.size[other]
            return True

        def find_expr(self, expr: sympy.Expr) -> sympy.Expr:
            parent = self.find(self.reverse_expressions[expr])
            return self.expressions[parent]

        def find(self, x: int) -> int:
            if self.leader[x] != x:
                self.leader[x] = self.find(self.leader[x])
            return self.leader[x]

        def choose_leader(self, a: int, b: int) -> tuple[int, int]:
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


@lru_cache(maxsize=1024)
def _sub_unbacked_exprs(shape_env: ShapeEnv, expr: sympy.Expr) -> sympy.Expr:
    """Substitute unbacked expressions with canonical equivalents.
    Used by optimization_hint to maximize consistency when hinting unbacked symbols.
    """
    replacements = _get_unbacked_replacements(shape_env)
    symbol_replacements = {
        k: v for k, v in replacements.items() if isinstance(k, sympy.Symbol)
    }
    expr_replacements = {
        k: v for k, v in replacements.items() if not isinstance(k, sympy.Symbol)
    }

    # consider making this threshold configurable
    sub_cnt_limit = 30
    sub_cnt = 0
    while sub_cnt < sub_cnt_limit:
        if expr_replacements:
            free_symbol_count = len(expr.free_symbols)
            if (
                free_symbol_count * len(expr_replacements)
                <= SYMPY_SUBS_MAX_SYMBOL_REPLACEMENT_PRODUCT
            ):
                new_expr = expr.subs(replacements)
            else:
                new_expr = _sympy_subs(expr, symbol_replacements)
                log.debug(
                    "Skipping %d expensive unbacked expression substitutions "
                    "over %d free symbols in optimization_hint",
                    len(expr_replacements),
                    free_symbol_count,
                )
        else:
            new_expr = _sympy_subs(expr, symbol_replacements)
        if new_expr == expr:
            break
        if len(new_expr.free_symbols) <= SYMPY_FACTOR_MAX_FREE_SYMBOLS:
            expr = sympy.factor(new_expr)
        else:
            expr = new_expr
        sub_cnt += 1
    else:
        log.warning("Substitution limit (%d) reached w/ %s", sub_cnt_limit, expr)

    expr = _sympy_subs(expr, shape_env.backed_var_to_val)
    expr = _sympy_subs(expr, shape_env.var_to_hint_override)
    return expr


def _hint_bounds_from_runtime_asserts(
    shape_env: ShapeEnv, s: sympy.Symbol
) -> tuple[int | None, int | None]:
    """Concrete ``(lower, upper)`` bounds for an unbacked symbol derived from the
    runtime asserts, with backed symbols replaced by their hint values.

    Relational invariants such as ``u0 <= s44`` live only in
    ``deferred_runtime_asserts`` -- ``var_to_range`` keeps the loose
    ``[0, unbacked_symint_fallback]``. Feeding these bounds into the size hint keeps
    it consistent with the graph: e.g. a reducing slice ``x[u0:]`` sized ``s44 - u0``
    cannot hint negative, because ``u0`` is capped at ``s44``'s hint instead of the
    generic fallback. Only linear single-symbol bounds (``c*s + k <= 0``) with
    concrete integer coefficients are used, so this never tightens past what the
    asserts prove; anything else is skipped.
    """
    lower: int | None = None
    upper: int | None = None
    backed = shape_env.backed_var_to_val
    # Read only the asserts keyed to this symbol (bounds on an unbacked symbol
    # only ever live there); guards constrain backed symbols and would just be a
    # wasted scan.
    try:
        for ra in shape_env.deferred_runtime_asserts.get(s, ()):
            ax = ra.expr
            if not isinstance(ax, sympy.Basic):
                continue
            # Substitute backed hints, so e.g. ``u0 <= s44`` becomes ``u0 <= 16``.
            ax = ax.xreplace(backed)
            if s not in ax.free_symbols:
                continue
            # Normalize the relation to ``g < 0`` (strict) or ``g <= 0``.
            if isinstance(ax, sympy.LessThan):
                g, strict = ax.lhs - ax.rhs, False
            elif isinstance(ax, sympy.StrictLessThan):
                g, strict = ax.lhs - ax.rhs, True
            elif isinstance(ax, sympy.GreaterThan):
                g, strict = ax.rhs - ax.lhs, False
            elif isinstance(ax, sympy.StrictGreaterThan):
                g, strict = ax.rhs - ax.lhs, True
            else:
                continue
            # Require g == c*s + k with concrete integer c, k (no other free syms).
            c = g.coeff(s, 1)
            k = g.coeff(s, 0)
            if g - (c * s + k) != 0 or not (c.is_Integer and k.is_Integer):
                continue
            c, k = int(c), int(k)
            if c == 0:
                continue
            thr = sympy.Rational(-k, c)  # c*s + k (<|<=) 0  ->  s (<|<=) thr, c>0
            if c > 0:
                bnd = int(sympy.ceiling(thr) - 1) if strict else int(sympy.floor(thr))
                upper = bnd if upper is None else min(upper, bnd)
            else:  # dividing by c<0 flips the inequality: s (>|>=) thr
                bnd = int(sympy.floor(thr) + 1) if strict else int(sympy.ceiling(thr))
                lower = bnd if lower is None else max(lower, bnd)
    except Exception:
        pass
    return lower, upper


def _symbol_hint_bounds(
    shape_env: ShapeEnv, s: sympy.Symbol
) -> tuple[int | None, int | None]:
    lower: int | None = None
    upper: int | None = None
    vr = shape_env.var_to_range.get(s, None)
    if vr is not None:
        if isinstance(vr.lower, (int, sympy.Integer)):
            lower = int(vr.lower)
        if isinstance(vr.upper, (int, sympy.Integer)):
            upper = int(vr.upper)
    ra_lower, ra_upper = _hint_bounds_from_runtime_asserts(shape_env, s)
    if ra_lower is not None:
        lower = ra_lower if lower is None else max(lower, ra_lower)
    if ra_upper is not None:
        upper = ra_upper if upper is None else min(upper, ra_upper)
    return lower, upper


def _clamp_fallback(fallback: int, lower: int | None, upper: int | None) -> int:
    if lower is not None:
        fallback = max(fallback, lower)
    if upper is not None:
        fallback = min(fallback, upper)
    return fallback


def _multi_symbol_linear_constraints(
    shape_env: ShapeEnv,
) -> list[tuple[dict[sympy.Symbol, int], int]]:
    """Runtime asserts of the form ``sum(c_i * u_i) + k <= 0`` over two or more
    unbacked symbols, after backed symbols and equalities are substituted.
    Single-symbol bounds are already handled by ``_symbol_hint_bounds``.
    """
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    constraints: list[tuple[dict[sympy.Symbol, int], int]] = []
    seen: set[sympy.Basic] = set()
    for asserts in shape_env.deferred_runtime_asserts.values():
        for ra in asserts:
            ax = ra.expr
            if ax in seen:
                continue
            seen.add(ax)
            if isinstance(ax, sympy.LessThan):
                g, strict = ax.lhs - ax.rhs, False
            elif isinstance(ax, sympy.StrictLessThan):
                g, strict = ax.lhs - ax.rhs, True
            elif isinstance(ax, sympy.GreaterThan):
                g, strict = ax.rhs - ax.lhs, False
            elif isinstance(ax, sympy.StrictGreaterThan):
                g, strict = ax.rhs - ax.lhs, True
            else:
                continue
            try:
                g = g.xreplace(shape_env.replacements)
                g = sympy.expand(_sub_unbacked_exprs(shape_env, g))
                syms = list(g.free_symbols)
                if len(syms) < 2 or len(free_unbacked_symbols(g)) != len(syms):
                    continue
                coeffs = {s: g.coeff(s, 1) for s in syms}
                k = g.subs(dict.fromkeys(syms, 0))
                if sympy.expand(g - sum(c * s for s, c in coeffs.items()) - k) != 0:
                    continue
                if not k.is_Integer or not all(c.is_Integer for c in coeffs.values()):
                    continue
            except Exception:
                continue
            int_coeffs = {s: int(c) for s, c in coeffs.items() if c != 0}
            if len(int_coeffs) < 2:
                continue
            # Integers: g < 0  <=>  g + 1 <= 0.
            constraints.append((int_coeffs, int(k) + (1 if strict else 0)))
    return constraints


def _divisibility_constraints(
    shape_env: ShapeEnv,
) -> dict[sympy.Symbol, tuple[int, int]]:
    """``Eq(Mod(u, m), r)`` runtime asserts on a single unbacked symbol, as
    ``u -> (m, r)``. Several zero-residue moduli on one symbol combine via lcm;
    other conflicting forms keep the first one seen.
    """
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
    from torch.utils._sympy.functions import Mod, PythonMod

    mod_types = (sympy.Mod, Mod, PythonMod)
    mods: dict[sympy.Symbol, tuple[int, int]] = {}
    for asserts in shape_env.deferred_runtime_asserts.values():
        for ra in asserts:
            ax = ra.expr
            if not isinstance(ax, sympy.Equality):
                continue
            lhs, rhs = ax.lhs, ax.rhs
            if isinstance(rhs, mod_types):
                lhs, rhs = rhs, lhs
            if not (isinstance(lhs, mod_types) and rhs.is_Integer):
                continue
            base, m = lhs.args
            if not (
                isinstance(base, sympy.Symbol)
                and m.is_Integer
                and int(m) > 1
                and free_unbacked_symbols(base)
            ):
                continue
            m, r = int(m), int(rhs) % int(m)
            prev = mods.get(base)
            if prev is None:
                mods[base] = (m, r)
            elif prev[1] == 0 and r == 0:
                mods[base] = (prev[0] * m // math.gcd(prev[0], m), 0)
    return mods


def _round_up(v: int, mod: tuple[int, int] | None) -> int:
    if mod is None:
        return v
    m, r = mod
    return v + (r - v) % m


def _round_down(v: int, mod: tuple[int, int] | None) -> int:
    if mod is None:
        return v
    m, r = mod
    return v - (v - r) % m


def _repair_hints(
    constraints: list[tuple[dict[sympy.Symbol, int], int]],
    assignment: dict[sympy.Symbol, int],
    bounds: dict[sympy.Symbol, tuple[int | None, int | None]],
    mods: dict[sympy.Symbol, tuple[int, int]],
) -> tuple[dict[sympy.Symbol, int], bool]:
    """Greedily adjust ``assignment`` until every constraint holds.

    For a violated ``sum(c_i * u_i) + k <= 0`` we first raise symbols with a
    negative coefficient (e.g. a backing-storage size) within their upper
    bound, then lower symbols with a positive coefficient (e.g. view size and
    offset) evenly within their lower bound. Every move lands on a value that
    keeps the symbol's ``Mod`` residue from ``mods``. Returns the assignment
    with the fewest violations seen and whether it satisfies every constraint.
    """

    def excess(coeffs: dict[sympy.Symbol, int], k: int) -> int:
        return sum(c * x[s] for s, c in coeffs.items()) + k

    def num_violations() -> int:
        return sum(1 for coeffs, k in constraints if excess(coeffs, k) > 0)

    def lowest(s: sympy.Symbol) -> int | None:
        lower = bounds[s][0]
        return None if lower is None else _round_up(lower, mods.get(s))

    x = dict(assignment)
    best, best_violations = dict(x), num_violations()
    for _ in range(CONSISTENT_HINTS_MAX_ITERS):
        if best_violations == 0:
            break
        for coeffs, k in constraints:
            e = excess(coeffs, k)
            if e <= 0:
                continue
            for s, c in coeffs.items():
                if c >= 0:
                    continue
                mod = mods.get(s)
                target = _round_up(x[s] - (e // c), mod)
                upper = bounds[s][1]
                if upper is not None and target > upper:
                    target = _round_down(upper, mod)
                if target > x[s]:
                    e -= (target - x[s]) * -c
                    x[s] = target
                if e <= 0:
                    break
            lowerable = [s for s, c in coeffs.items() if c > 0]
            while e > 0 and lowerable:
                share = -(-e // len(lowerable))
                still_lowerable = []
                for s in lowerable:
                    c = coeffs[s]
                    mod = mods.get(s)
                    target = _round_down(x[s] - -(-share // c), mod)
                    floor = lowest(s)
                    if floor is not None and target < floor:
                        target = floor
                    if target < x[s]:
                        e -= (x[s] - target) * c
                        x[s] = target
                        if floor is None or target > floor:
                            still_lowerable.append(s)
                    if e <= 0:
                        break
                lowerable = still_lowerable
        violations = num_violations()
        if violations < best_violations:
            best, best_violations = dict(x), violations
        else:
            # No progress this pass; further passes repeat the same moves.
            break
    return best, best_violations == 0


def _get_consistent_unbacked_hints(
    shape_env: ShapeEnv, fallback: int
) -> tuple[dict[sympy.Symbol, int], bool]:
    """One hint per unbacked symbol, chosen jointly so that multi-symbol linear
    runtime asserts (e.g. ``u_offset + u_size <= u_storage``) hold. Hinting each
    symbol independently can violate them, which produces example tensors
    whose views do not fit their backing storage.

    Returns the assignment and whether every collected assert is satisfied.
    Symbols not mentioned by such asserts are omitted.
    """
    key = (fallback, shape_env._version_counter, len(shape_env.var_to_hint_override))
    cached = shape_env._consistent_unbacked_hints
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]

    constraints = _multi_symbol_linear_constraints(shape_env)
    mods = _divisibility_constraints(shape_env)
    assignment: dict[sympy.Symbol, int] = {}
    bounds: dict[sympy.Symbol, tuple[int | None, int | None]] = {}
    symbols = [s for coeffs, _ in constraints for s in coeffs] + list(mods)
    for s in symbols:
        if s in bounds:
            continue
        bounds[s] = lower, upper = _symbol_hint_bounds(shape_env, s)
        v = _round_down(_clamp_fallback(fallback, lower, upper), mods.get(s))
        if lower is not None and v < lower:
            v = _round_up(lower, mods.get(s))
        assignment[s] = v

    consistent = True
    if constraints:
        assignment, consistent = _repair_hints(constraints, assignment, bounds, mods)
        if not consistent:
            log.debug(
                "Could not find unbacked hints satisfying all linear asserts: %s",
                assignment,
            )

    shape_env._consistent_unbacked_hints = (key, assignment, consistent)
    return assignment, consistent


def unbacked_hints_consistent(shape_env: ShapeEnv, fallback: int | None = None) -> bool:
    """Whether optimization hints for unbacked symbols satisfy every
    multi-symbol linear runtime assert."""
    if fallback is None:
        from torch._inductor.config import unbacked_symint_fallback

        fallback = unbacked_symint_fallback
    return _get_consistent_unbacked_hints(shape_env, fallback)[1]


def _optimization_hint_base(
    shape_env: ShapeEnv,
    expr: sympy.Expr | int,
    precomputed_replacements: dict[sympy.Expr, sympy.Symbol],
    fallback: int | None = None,
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

    # to have expanded (Identity free) expr stored in original
    if isinstance(expr, sympy.Expr):
        expr = expr.expand(identity=True)

    original = expr
    # sympy.expand() doesn't work with boolean expressions like Or/And
    if isinstance(expr, sympy.Expr):
        expr = expr.xreplace(shape_env.replacements)
    else:
        expr = sympy.sympify(expr).xreplace(shape_env.replacements)

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
        expr = _sympy_subs(expr, precomputed_replacements)

    expr = _sympy_subs(expr, shape_env.backed_var_to_val)
    if isinstance(expr, sympy.Expr):
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
        if (
            isinstance(original, sympy.Expr)
            and len(original.free_symbols) <= SYMPY_FACTOR_MAX_FREE_SYMBOLS
        ):
            original = sympy.factor(original)
        expr = _sub_unbacked_exprs(shape_env, original)

    # For multiple expressions that depend on an unbacked symint,
    # we want to compute them consistently for a size hint we have chosen.
    # So, recursively compute expressions via size hints of contained symbols.
    # For example: u1 * u2 - 10 ==> fallback * fallback - 10

    if not isinstance(expr, sympy.Expr):
        raise RuntimeError(f"Expected sympy Expr, got {type(expr)}: {expr}")
    free_symbols = expr.free_symbols

    # Constrain fallback per-symbol based on var_to_range bounds, then tighten
    # further with invariants that live only in the runtime asserts (e.g.
    # u0 <= s44). Without the latter, an unbacked symbol keeps its generic
    # fallback even when the graph proves it is bounded by a backed dim, so a
    # size expression like s44 - u0 can hint negative and overflow downstream
    # allocations (autotuning example tensors, buffer sizing).
    size_dict = {}
    joint_hints, _ = _get_consistent_unbacked_hints(shape_env, fallback)
    for s in free_symbols:
        if s in joint_hints:
            size_dict[s] = joint_hints[s]
            continue
        size_dict[s] = _clamp_fallback(fallback, *_symbol_hint_bounds(shape_env, s))

    try:
        final_result = _sympy_subs(expr, size_dict)
    except ZeroDivisionError:
        # Expressions like ModularIndexing(x, u1, 4) crash during substitution
        # when u1 is replaced with 0, because sympy eagerly evaluates
        # (x // 0) % 4.  This can happen when an unbacked symbol with
        # var_to_range lower=0 is used as a divisor (e.g. from
        # _dynamic_reshape_indexer) and the fallback also maps to 0.
        # Return fallback in that case.
        return fallback if fallback is not None else 0

    final_result = _maybe_realize_expr(final_result, fallback)
    if final_result is None:
        raise RuntimeError(f"Failed to realize expression to int: {expr}")

    return final_result
