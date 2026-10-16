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

import contextlib
import dataclasses
import logging
import math
import sys
import threading
from functools import lru_cache
from typing import Any, NamedTuple, TYPE_CHECKING

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

# Inside consistent_hints, equal shares of an Eq's slack are rounded down to a
# multiple of this, so the parts keep common alignments.
_FAIR_SHARE_ALIGNMENT = 8

if TYPE_CHECKING:
    from collections.abc import Iterator

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


def _runtime_asserts_by_symbol(
    shape_env: ShapeEnv,
) -> dict[sympy.Symbol, list[sympy.Basic]]:
    """Map each symbol to the runtime asserts that mention it, with
    replacements applied. ``deferred_runtime_asserts`` keys each assert by its
    newest unbacked symbol only.
    """
    from collections import defaultdict

    by_symbol: dict[sympy.Symbol, list[sympy.Basic]] = defaultdict(list)
    for asserts in shape_env.deferred_runtime_asserts.values():
        for ra in asserts:
            if not isinstance(ra.expr, sympy.Basic):
                continue
            expr = ra.expr.xreplace(shape_env.replacements)
            for sym in expr.free_symbols:
                by_symbol[sym].append(expr)
    return by_symbol


@dataclasses.dataclass
class ConsistentHintState:
    """While set by consistent_hints(state), computes optimization hints for
    shape_env from one shared per-symbol assignment.

    Use it when several hints must be consistent, e.g. the example inputs of
    compile-time autotuning, where a view must fit its backing storage and
    scalar arguments must match tensor shapes (see
    PythonWrapperCodegen.generate_example_arg_value). The caller keeps the
    state, so values chosen in one use of the context are reused by the next.

    While set, optimization_hint(expr) canonicalizes expr as it does outside
    the context, then gives each remaining unbacked symbol without a value one,
    oldest first: the values already in assignments are substituted into the
    runtime asserts, which then bound it. Its value is stored in assignments
    and reused by every later hint, and expr is evaluated with those values. E.g. with ``u0 + u1 <= u2`` and fallback 100:

        optimization_hint(u0 + u1)   # decomposed into u0 and u1
            u0 -> 100
            u1 -> 100    (u2 still unknown, so the assert gives no bound)
            returns 200
        optimization_hint(u2)
            u2 -> 200    (the assert becomes 100 + 100 <= u2)
            returns 200

    Outside the context each symbol gets 100 independently, so u0 + u1 = 200
    but u2 = 100, which violates the assert.

    This is a best-effort heuristic that uses the existing per-symbol hinting
    logic, and tracks the values already chosen and substitutes them into the
    runtime asserts. The consistency comes from fixing each symbol's value once
    it is chosen: if u0 was hinted as 100, every subsequent optimization_hint
    call uses u0 = 100, both when evaluating an expression and when bounding
    other symbols through the asserts. It does not make every hint consistent
    with every assert (e.g. non-linear or coupled asserts can still be
    violated), but it covers the practical cases, such as a view's offset and
    size within its storage. As a fallback, users can set an unbacked symbol's
    hint explicitly with torch._dynamo.override_optimization_hint.
    """

    shape_env: ShapeEnv
    # Unbacked symbol -> value chosen for it.
    assignments: dict[sympy.Symbol, int] = dataclasses.field(default_factory=dict)
    # Symbol -> runtime asserts mentioning it, for shape_env._version_counter
    # == version.
    asserts_by_symbol: dict[sympy.Symbol, list[sympy.Basic]] = dataclasses.field(
        default_factory=dict
    )
    version: int | None = None


_consistent_hints_tls = threading.local()


def _current_consistent_hints() -> ConsistentHintState | None:
    return getattr(_consistent_hints_tls, "state", None)


@contextlib.contextmanager
def consistent_hints(state: ConsistentHintState) -> Iterator[None]:
    """Compute optimization hints for state.shape_env from one shared
    per-symbol assignment; see ConsistentHintState."""
    shape_env = state.shape_env
    if state.version != shape_env._version_counter:
        state.asserts_by_symbol = _runtime_asserts_by_symbol(shape_env)
        state.version = shape_env._version_counter
    prev = getattr(_consistent_hints_tls, "state", None)
    _consistent_hints_tls.state = state
    try:
        yield
    finally:
        _consistent_hints_tls.state = prev


class _LinearBound(NamedTuple):
    # c*s + k <= 0, with every other unknown symbol at its chosen range end.
    c: int
    k: int
    # Other unknown symbols with a positive coefficient: the parts that share
    # the bound with s (b in ``a + b - 100 <= 0``).
    n_parts: int
    # gcd of the coefficients of all other unknown symbols. For an Eq to have
    # an integer solution, c*s + k must be a multiple of it.
    peers_gcd: int
    # sum over the parts t of c_t * (min(fallback, t's max) - t's min): how much
    # more the parts take if each gets its fallback instead of its minimum.
    parts_fallback: int


def _implied_linear_bound(
    shape_env: ShapeEnv,
    g: sympy.Expr,
    s: sympy.Symbol,
    fallback: int | None = None,
) -> _LinearBound | None:
    """Rewrite the runtime assert ``g <= 0`` as ``c*s + k <= 0``, where s is
    the symbol being bounded, by replacing every other symbol t in g with a
    number: the end of t's range that gives s the most room (t's minimum if its
    coefficient is positive, its maximum if negative). E.g. ``a + b - 100 <= 0``
    with b in [2, 50] gives c = 1, k = -98, i.e. a <= 98.

    This is sound: the chosen end makes each ``c_t * t`` as small as it can be,
    so ``c*s + k <= g`` for every t in its range, and any s that satisfies the
    assert (g <= 0) for some t also satisfies ``c*s + k <= 0``. It is also
    exact for a single assert, since t at that end is itself a valid choice.

    Returns None if g is not linear with integer coefficients or a needed range
    end is infinite. See _LinearBound for the other fields.
    """
    c = 0
    k = 0
    n_parts = 0
    peers_gcd = 0
    parts_fallback = 0
    for term, coeff in sympy.expand(g).as_coefficients_dict().items():
        if not coeff.is_Integer:
            return None
        coeff = int(coeff)
        if term == 1:
            k += coeff
        elif term == s:
            c += coeff
        elif isinstance(term, sympy.Symbol):
            vr = shape_env.var_to_range.get(term)
            if vr is None:
                return None
            bound = vr.lower if coeff > 0 else vr.upper
            if not isinstance(bound, (int, sympy.Integer)):
                return None
            k += coeff * int(bound)
            peers_gcd = math.gcd(peers_gcd, abs(coeff))
            if coeff > 0:
                n_parts += 1
                if fallback is not None:
                    top = fallback
                    if isinstance(vr.upper, (int, sympy.Integer)):
                        top = min(top, int(vr.upper))
                    parts_fallback += coeff * max(0, top - int(bound))
        else:
            return None
    return _LinearBound(c, k, n_parts, peers_gcd, parts_fallback)


def _round_for_integer_leftover(x: int, c: int, k: int, g: int, lo: int) -> int:
    """Largest s <= x with s >= lo such that ``c*s + k`` is a multiple of g, so
    the remaining parts of an Eq can still add up exactly (e.g. leave a
    multiple of 96 for a part with coefficient 96); x if there is none."""
    if g <= 1:
        return x
    d = math.gcd(c, g)
    if k % d:
        return x
    m = g // d
    r = (-k // d) * pow(c // d, -1, m) % m
    s = x - (x - r) % m
    return s if s >= lo else x


def _unbacked_hint_bounds_from_runtime_asserts(
    shape_env: ShapeEnv,
    s: sympy.Symbol,
    state: ConsistentHintState | None = None,
    fallback: int | None = None,
) -> tuple[int | None, int | None]:
    """Concrete ``(lower, upper)`` bounds for an unbacked symbol derived from the
    runtime asserts, with backed symbols replaced by their hint values.

    Relational invariants such as ``u0 <= s44`` live only in
    ``deferred_runtime_asserts`` -- ``var_to_range`` keeps the loose
    ``[0, unbacked_symint_fallback]``. Feeding these bounds into the size hint keeps
    it consistent with the graph: e.g. a reducing slice ``x[u0:]`` sized ``s44 - u0``
    cannot hint negative, because ``u0`` is capped at ``s44``'s hint instead of the
    generic fallback.

    Each assert mentioning s is used in three steps:

    1. Normalize it to ``g <= 0`` (or ``g < 0``); an Eq gives two relations,
       ``lhs - rhs <= 0`` and ``rhs - lhs <= 0``. Only g linear with integer
       coefficients is used; anything else (products, Mod, FloorDiv, rational
       coefficients) is skipped.
    2. Rewrite it as ``c*s + k <= 0`` by replacing the other unknown symbols
       with numbers (see _implied_linear_bound).
    3. Solve the remaining ``c*s + k <= 0`` for s: ``c > 0`` gives the upper
       bound ``s <= -k/c``, ``c < 0`` the lower bound ``s >= -k/c``.
    """
    lower: int | None = None
    upper: int | None = None
    backed = shape_env.backed_var_to_val
    known_assignments: dict[sympy.Symbol, int] = (
        state.assignments if state is not None else {}
    )
    if state is not None:
        # Any assert mentioning s can bound it, through the known values or the
        # ranges of its other symbols, so read every such assert, not only
        # those keyed to s.
        exprs = state.asserts_by_symbol.get(s, ())
    else:
        # Outside the context, read only the asserts keyed to s (those whose
        # newest unbacked symbol is s), as before; finding every assert that
        # mentions s needs the index the context builds.
        # TODO: explore reading every assert that mentions s here as well.
        exprs = [ra.expr for ra in shape_env.deferred_runtime_asserts.get(s, ())]
    try:
        for ax in exprs:
            if not isinstance(ax, sympy.Basic):
                continue
            # Substitute backed hints, hint overrides and known assignments, in
            # the order _optimization_hint_impl uses, so e.g. ``u0 <= s44``
            # becomes ``u0 <= 16``.
            ax = (
                ax.xreplace(backed)
                .xreplace(shape_env.var_to_hint_override)
                .xreplace(known_assignments)
            )
            if s not in ax.free_symbols:
                continue
            # Normalize the relation to ``g < 0`` (strict) or ``g <= 0``.
            if isinstance(ax, sympy.LessThan):
                relations = [(ax.lhs - ax.rhs, False)]
            elif isinstance(ax, sympy.StrictLessThan):
                relations = [(ax.lhs - ax.rhs, True)]
            elif isinstance(ax, sympy.GreaterThan):
                relations = [(ax.rhs - ax.lhs, False)]
            elif isinstance(ax, sympy.StrictGreaterThan):
                relations = [(ax.rhs - ax.lhs, True)]
            elif isinstance(ax, sympy.Equality):
                # ``a == b`` is ``a - b <= 0`` and ``b - a <= 0``; e.g. split's
                # ``Eq(u0 + u1, 100)`` pins u1 to ``100 - u0`` once u0 is known.
                relations = [(ax.lhs - ax.rhs, False), (ax.rhs - ax.lhs, False)]
            else:
                continue
            is_eq = isinstance(ax, sympy.Equality)
            for g, strict in relations:
                linear = _implied_linear_bound(shape_env, g, s, fallback)
                if linear is None:
                    continue
                c, k = linear.c, linear.k
                if c == 0:
                    continue
                # The parts of an Eq share a fixed budget; inside the context,
                # spread it over the parts that are still unknown.
                fair = state is not None and is_eq and linear.n_parts > 0
                if fair and c < 0:
                    # s is the total and some parts are still unknown: leave
                    # each its fallback, so they are not squeezed to 0. E.g.
                    # Eq(96*u0 + 64*u1, u2) with u0 known gives
                    # u2 = 96*u0 + 64*fallback instead of 96*u0 and u1 = 0.
                    k += linear.parts_fallback
                thr = sympy.Rational(-k, c)  # c*s + k (<|<=) 0  ->  s (<|<=) thr, c>0
                if c > 0:
                    bnd = (
                        int(sympy.ceiling(thr) - 1) if strict else int(sympy.floor(thr))
                    )
                    if fair:
                        # s is a part: take an equal share of the slack,
                        # rounded down to a multiple of _FAIR_SHARE_ALIGNMENT
                        # so common alignment checks (% 8) still hold, and so
                        # the remaining parts can still add up exactly; the
                        # last part takes the remainder. E.g. Eq(u0 + u1 + u2 +
                        # u3, 100) gives 24, 24, 24, 28 instead of 100 for the
                        # first part hinted and 0 for the rest.
                        vr = shape_env.var_to_range.get(s)
                        s_min = (
                            int(vr.lower)
                            if vr is not None
                            and isinstance(vr.lower, (int, sympy.Integer))
                            else 0
                        )
                        share = s_min + max(0, bnd - s_min) // (linear.n_parts + 1)
                        aligned = share - share % _FAIR_SHARE_ALIGNMENT
                        target = aligned if aligned >= max(s_min, 1) else share
                        if fallback is not None:
                            # s gets min(fallback, its bound), so round that.
                            target = min(target, fallback)
                        bnd = _round_for_integer_leftover(
                            target, c, k, linear.peers_gcd, s_min
                        )
                    upper = bnd if upper is None else min(upper, bnd)
                else:  # dividing by c<0 flips the inequality: s (>|>=) thr
                    bnd = (
                        int(sympy.floor(thr) + 1) if strict else int(sympy.ceiling(thr))
                    )
                    lower = bnd if lower is None else max(lower, bnd)
    except Exception:
        pass
    return lower, upper


def _bounded_symbol_hint(
    shape_env: ShapeEnv,
    s: sympy.Symbol,
    fallback: int,
    state: ConsistentHintState | None,
) -> int:
    """fallback clamped to s's bounds from var_to_range and the runtime
    asserts."""
    lower: int | None = None
    upper: int | None = None
    vr = shape_env.var_to_range.get(s, None)
    if vr is not None:
        if isinstance(vr.lower, (int, sympy.Integer)):
            lower = int(vr.lower)
        if isinstance(vr.upper, (int, sympy.Integer)):
            upper = int(vr.upper)
    ra_lower, ra_upper = _unbacked_hint_bounds_from_runtime_asserts(
        shape_env, s, state, fallback
    )
    if ra_lower is not None:
        lower = ra_lower if lower is None else max(lower, ra_lower)
    if ra_upper is not None:
        upper = ra_upper if upper is None else min(upper, ra_upper)
    sym_fallback = fallback
    if lower is not None:
        sym_fallback = max(sym_fallback, lower)
    if upper is not None:
        sym_fallback = min(sym_fallback, upper)
    return sym_fallback


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

    Inside consistent_hints(), hints are computed from one shared per-symbol
    assignment; see ConsistentHintState.

    Args:
        shape_env: The ShapeEnv instance.
        expr: A sympy expression or integer to evaluate.
        precomputed_replacements: Precomputed replacements for PRECOMPUTED_SIZE symbols.
        fallback: Fallback value for unbacked symbols. If None, reads from config.

    Returns:
        A concrete integer hint for the expression.
    """
    return _optimization_hint_impl(
        shape_env,
        expr,
        precomputed_replacements,
        fallback,
        _current_consistent_hints(),
    )


def _optimization_hint_impl(
    shape_env: ShapeEnv,
    expr: sympy.Expr | int,
    precomputed_replacements: dict[sympy.Expr, sympy.Symbol],
    fallback: int | None,
    state: ConsistentHintState | None,
) -> int:
    """_optimization_hint_base for one expression. With state, each unbacked
    symbol left after canonicalization that has no value yet gets one, oldest
    first, stored in state.assignments; see ConsistentHintState."""
    from torch.fx.experimental.symbolic_shapes import (
        has_free_unbacked_symbols,
        symbol_is_type,
        SymT,
    )

    # Read config at call time to respect runtime patches (e.g., in tests)
    if fallback is None:
        from torch._inductor.config import unbacked_symint_fallback

        fallback = unbacked_symint_fallback
    known_assignments: dict[sympy.Symbol, int] = (
        state.assignments if state is not None else {}
    )

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
        # Substitute known values only after the Eq canonicalization above, so
        # e.g. u0 * u1 still becomes 100 given Eq(u0 * u1, 100).
        expr = _sympy_subs(expr, known_assignments)
        result = _maybe_realize_expr(expr, fallback)
        if result is not None:
            return result

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
    if state is None:
        for s in free_symbols:
            size_dict[s] = _bounded_symbol_hint(shape_env, s, fallback, None)
    else:
        # Give each remaining unbacked symbol a value once, oldest first, so
        # later symbols are bounded by earlier ones; every subsequent hint
        # reuses it. For u<idx> names, sorting by (length, name) is creation
        # order.
        for s in sorted(free_symbols, key=lambda s: (len(s.name), s.name)):
            if not symbol_is_type(s, SymT.UNBACKED_INT):
                size_dict[s] = _bounded_symbol_hint(shape_env, s, fallback, state)
                continue
            if s not in state.assignments:
                state.assignments[s] = _bounded_symbol_hint(
                    shape_env, s, fallback, state
                )
            size_dict[s] = state.assignments[s]

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
