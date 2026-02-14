# mypy: allow-untyped-defs
import functools
import itertools
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast, Optional, Union

import sympy
from sympy import Expr

from torch import SymInt
from torch.fx.experimental.symbolic_shapes import (
    free_symbols,
    has_free_unbacked_symbols,
    ShapeEnv,
)
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.symbol import symbol_is_type, SymT
from torch.utils._sympy.value_ranges import bound_sympy, IntInfinity, ValueRanges

from . import config
from .runtime.runtime_utils import is_power_of_2
from .utils import (
    has_free_symbols,
    sympy_index_symbol,
    sympy_index_symbol_with_prefix,
    sympy_subs,
    VarRanges,
)
from .virtualized import V


log = logging.getLogger(__name__)


def statically_known_true(
    shape_env: ShapeEnv,
    expr: Union[sympy.Basic, bool],
    axioms: Optional[tuple[sympy.Expr]] = None,
    var_to_range: Optional[tuple[tuple[sympy.Symbol, ValueRanges[Any]]]] = None,
) -> bool:
    if expr in (True, False):
        return bool(expr)

    try:
        simplified = shape_env._maybe_evaluate_static(
            expr,
            axioms=axioms,
            var_to_range=var_to_range,
        )
        if simplified is not None:
            return bool(simplified)
    except Exception:
        log.debug("Could not simplify  %s", expr, exc_info=True)

    return False


# This class is a little awkward, because ShapeEnv is doing most of the heavy
# lifting and in some cases we should be directly passing through to ShapeEnv,
# but there is some extra inductor logic that needs to be handled here
class SizeVarAllocator:
    """
    A class that manages symbolic size variables and their relationships.

    This class works with the ShapeEnv to handle symbolic shape expressions,
    simplify them, and provide utilities for guarding, checking, and evaluating
    symbolic expressions. It also manages precomputed replacements and stride
    calculations for tensor operations.
    """

    def __init__(self, shape_env=None) -> None:
        super().__init__()
        # Note: this can lead to bugs. Reasoning APIs depends on existing information in
        # in the shape_env. For example! var_to_ranges can't be empty!
        if shape_env is None:
            shape_env = ShapeEnv()
        self.shape_env = shape_env
        self.backed_var_to_val = self.shape_env.backed_var_to_val
        self.var_to_hint_override = self.shape_env.var_to_hint_override
        self.replacements: dict[sympy.Symbol, Expr] = self.shape_env.replacements
        self.unbacked_replacements: Optional[dict[Expr, Expr]] = None
        # Maps of dynamic sizes that have to be precomputed on the host to the kernel args.
        # The basic idea is if we have some complicated sympy expression
        # f(s0), we may choose to precompute it on the host and then replace
        # all occurrences of that sympy expression with ps0, so that when we
        # codegen we simply reference ps0 directly without repeating
        # f(s0).  Unlike regular size variables, ps variables cannot be
        # guarded upon; so if we are asked to guard on a Sympy expression
        # which potentially could have already had a precomputed replacement
        # on it, we are obligated to invert the precomputed replacements
        # (inv_precomputed_replacements).
        self.precomputed_replacements: dict[Expr, sympy.Symbol] = {}
        self.inv_precomputed_replacements: dict[sympy.Symbol, Expr] = {}
        self.stride_vars = self.make_stride_vars_cache()
        self.simplify_with_ranges = self.make_simplify_with_ranges_cache()
        self._simplify_loops = self.make_simplify_loops_cache()

    def simplify(self, expr: Expr):
        return sympy.expand(expr).xreplace(self.replacements)

    def make_simplify_with_ranges_cache(self) -> Callable[[Expr, VarRanges], Expr]:
        """
        self._simplify_with_ranges() can be expensive, cache its results
        """
        cache: dict[tuple[Any, ...], Expr] = {}
        replacement_count = len(self.replacements)

        def simplify_with_ranges(expr: Expr, var_ranges: VarRanges) -> Expr:
            nonlocal replacement_count
            if replacement_count != len(self.replacements):
                # new replacements invalidates cached results
                cache.clear()
                replacement_count = len(self.replacements)
            key = (expr, *var_ranges.items())
            result = cache.get(key)
            if result is None:
                result = self._simplify_with_ranges(expr, var_ranges)
                cache[key] = result
                if result != expr:
                    cache[(result, *var_ranges.items())] = result
            return result

        return simplify_with_ranges

    def make_simplify_loops_cache(self):
        """
        self._simplify_with_ranges() can be expensive, cache its results
        """
        cache: dict[tuple[Any, ...], Any] = {}
        replacement_count = len(self.replacements)

        def simplify_loops(index_vars, sizes, index_formulas):
            nonlocal replacement_count
            if replacement_count != len(self.replacements):
                # new replacements invalidates cached results
                cache.clear()
                replacement_count = len(self.replacements)
            key = (*index_vars, *sizes, *index_formulas)
            result = cache.get(key)
            if result is None:
                result = self._simplify_loops_impl(index_vars, sizes, index_formulas)
                cache[key] = result
            return result

        return simplify_loops

    def _simplify_with_ranges(self, expr: Expr, var_ranges: VarRanges) -> Expr:
        """
        Simplify indexing expression with knowledge of the ranges of
        iteration variables.
        """

        expr = join_dimensions(self.simplify(expr))
        original_expr = expr

        var_to_range = dict(self.shape_env.var_to_range)
        var_to_range.update(
            {
                k: ValueRanges(
                    0, max(0, v - 1) if not has_free_symbols([v]) else IntInfinity()
                )
                for k, v in var_ranges.items()
            }
        )
        for var in expr.free_symbols:
            if var not in var_to_range:
                var_to_range[var] = ValueRanges(0, IntInfinity())

        var_to_range_tuple = cast(
            tuple[tuple[sympy.Symbol, ValueRanges[sympy.Expr]]],
            tuple(var_to_range.items()),
        )

        axioms = []
        for var, upper_bound in var_ranges.items():
            axioms.append(0 <= var)
            axioms.append(var < upper_bound)
        axioms = tuple(axioms) + self.shape_env.get_axioms()

        def statically_known(expr):
            evaluated = self.shape_env._maybe_evaluate_static(
                expr,
                # pyrefly: ignore [bad-argument-type]
                axioms=axioms,
                var_to_range=var_to_range_tuple,
            )
            return bool(evaluated)

        def remove_zero_terms(base, divisor):
            """Symbols smaller than the divisor are zero"""
            if not statically_known(base >= 0):
                return base

            for v in base.free_symbols:
                if v in var_ranges:
                    # var smaller than divisor can be removed
                    # if the rest is guaranteed to be multiple of divisor
                    rest = sympy.Wild("_rest", exclude=[v])
                    m = base.match(v + rest)
                    if m and v not in m[rest].free_symbols:
                        gcd = sympy.gcd(m[rest], divisor)
                        if gcd == divisor:
                            if statically_known(v < divisor):
                                base = m[rest]
            return base

        def visit_indexing_div(base, divisor):
            return FloorDiv(remove_zero_terms(base, divisor), divisor)

        def visit_modular_indexing(base, divisor, modulus):
            base = remove_zero_terms(base, divisor)

            can_remove_mod = statically_known(base >= 0) and statically_known(
                base < modulus * divisor
            )

            if can_remove_mod:
                return FloorDiv(base, divisor)
            return ModularIndexing(base, divisor, modulus)

        if expr.has(ModularIndexing):
            expr = expr.replace(
                ModularIndexing(
                    sympy.Wild("base", integer=True),
                    sympy.Wild("divisor", integer=True),
                    sympy.Wild("modulus", integer=True),
                ),
                visit_modular_indexing,
            )

        if expr.has(FloorDiv):
            expr = expr.replace(
                FloorDiv(
                    sympy.Wild("base", integer=True),
                    sympy.Wild("divisor", integer=True),
                ),
                visit_indexing_div,
            )

        if expr != original_expr:
            return self._simplify_with_ranges(expr, var_ranges)
        return expr

    def _simplify_loops_impl(
        self, index_vars: list[sympy.Symbol], sizes, index_formulas
    ):
        """
        Try to remove as many axis from loop iterations as possible, by:
            1) removing size==1 dimensions
            2) fuse contiguous dimensions into a single loop
            If channel_last = True, we will prevent the last dim fused with other dims
        """
        sizes = list(map(self.simplify, sizes))

        strides = [
            # index_formulas may contain boolean expressions (e.g. s0 < 10),
            # for which "strides" don't make sense so we ignore them here.
            # NOTE: These expressions may still block merging dims in the sound
            # substitution test performed in can_merge_dims.
            (
                self.stride_vars(x, index_vars)
                if isinstance(x, sympy.Expr)
                else [0] * len(index_vars)
            )
            for x in index_formulas
        ]
        assert len(sizes) == len(strides[0]), (len(sizes), len(strides[0]))

        for i in range(len(sizes)):
            if sizes[i] == 1:
                # remove dim
                sizes[i] = None

        def can_merge_dims(a, b):
            for k in range(len(strides)):
                if self.simplify(strides[k][a] * sizes[a]) == self.simplify(
                    strides[k][b]
                ):
                    # approximate test passed, try sound version
                    va = index_vars[a]
                    vb = index_vars[b]
                    m1 = sympy_index_symbol("_merge_tester1")
                    m2 = sympy_index_symbol("_merge_tester2")
                    # NOTE: can't sub vb=0 here in case va * vb appears in the expression,
                    # in which case both expr1 and expr2 would be zero!
                    expr1 = sympy_subs(index_formulas[k], {va: m1 * sizes[a], vb: m2})
                    expr2 = sympy_subs(index_formulas[k], {va: 0, vb: (m1 + m2)})
                    if self.simplify(expr1) == self.simplify(expr2):
                        continue
                return False
            return True

        changed = True
        while changed:
            changed = False
            for i, j in itertools.product(
                reversed(range(len(sizes))), reversed(range(len(sizes)))
            ):
                if i == j or sizes[i] is None or sizes[j] is None:
                    continue
                if can_merge_dims(i, j):
                    changed = True
                    sizes[i] = sizes[i] * sizes[j]
                    sizes[j] = None

        def reindex(index):
            it = list(reversed(index))
            new_index = []
            for size in sizes:
                if size is None:
                    new_index.append(sympy.S.Zero)
                else:
                    new_index.append(it.pop())
            assert not it
            return new_index

        def prune(index):
            assert len(index) == len(sizes)
            return [i for i, s in zip(index, sizes) if s is not None]

        return [x for x in sizes if x is not None], reindex, prune

    # Note - [On Statically Known]
    # The statically_known_* family of functions below NEVER guard, they could return True if the
    # asked questions can be answered without guarding otherwise they return False.
    # Those are similar to statically_known_true in symbolic_shapes.py but operate on sympy
    # expressions instead of symnodes.
    def statically_known_true(self, expr: Union[sympy.Basic, bool]) -> bool:
        """
        Returns true if an expression is always true (symbolically or via guards),
        false otherwise. Never add guards, or throw data dependent errors.
        """
        return statically_known_true(self.shape_env, expr)

    def statically_known_equals(
        self, left: Union[Expr, int], right: Union[Expr, int]
    ) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left and right are equal.
        """
        return self.statically_known_true(sympy.Eq(left, right))  # type: ignore[arg-type]

    def statically_known_list_equals(
        self, left: Sequence[Expr], right: Sequence[Expr]
    ) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left and right lists are equal.
        """
        return len(left) == len(right) and all(
            self.statically_known_equals(l, r) for l, r in zip(left, right)
        )

    def statically_known_leq(self, left: Expr, right: Union[Expr, int]) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is less than or equal to right.
        """
        expr = left <= right
        return self.statically_known_true(expr)

    def statically_known_geq(self, left: Expr, right: Union[Expr, int]) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is greater than or equal to right.
        """
        expr = left >= right
        return self.statically_known_true(expr)

    def statically_known_lt(self, left: Expr, right: Union[Expr, int]) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is less than right.
        """
        expr = left < right
        return self.statically_known_true(expr)

    def statically_known_gt(self, left: Expr, right: Union[Expr, int]) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is greater than right.
        """
        expr = left > right
        return self.statically_known_true(expr)

    def statically_known_multiple_of(
        self, numerator: Expr, denominator: Union[Expr, int]
    ) -> bool:
        """
        Return a bool indicating if it is sound to optimize for the numerator being a multiple of the denominator.
        """
        # The reason we skip compute here is to avoid the cost of trying to eval this symbolically.
        # see https://github.com/sympy/sympy/issues/28200

        if len(free_symbols(numerator)) > 20:
            return False

        expr = sympy.Eq(numerator % denominator, 0)
        return self.statically_known_true(expr)  # type: ignore[arg-type]

    def statically_known_power_of_2(self, expr: Expr) -> bool:
        """
        Returns a bool indicating if x is known to be a power of 2.
        """
        return isinstance(expr, sympy.Integer) and is_power_of_2(int(expr))

    # The expect/check functions require you to ALREADY KNOW that a particular
    # condition holds. They are similar to expect_true in symbolic_shapes.py and
    # torch.check but operates on sympy expressions instead of symnodes.
    def expect_true(self, expr: Expr) -> bool:
        """
        Use it when you already know that expr is true or should be true and want to
        ensure that guards/runtime assertions are in place to ensure this in compiled
        function. Unlike check, this WON'T raise an error if expr isn't actually true.
        check Note [expect_true].
        """
        if not self.statically_known_true(expr):
            return self.shape_env.guard_or_defer_runtime_assert(
                expr, "sizevars.expect_true"
            )
        return True

    def check(self, expr: Expr) -> None:
        """
        Use it when you already know that expr is true or should be true and want to
        ensure that guards/runtime assertions are in place to ensure this in compiled
        function. Unlike expect_true, this WILL raise an error if expr isn't actually true.
        check Note [expect_true].
        """
        expr = sympy_subs(expr, self.inv_precomputed_replacements)
        assert self.expect_true(expr)

    def check_equals(self, left: Expr, right: Expr) -> None:
        """
        check(sympy.Eq(left, right)).

        """
        self.check(sympy.Eq(left, right))
        return left

    def check_equals_and_simplify(self, left: Expr, right: Expr) -> Expr:
        """
        check(sympy.Eq(left, right)) and returns left after applying
        inv_precomputed_replacements.
        """
        self.check(sympy.Eq(left, right))
        return sympy_subs(left, self.inv_precomputed_replacements)

    def check_leq(self, left: Expr, right: Expr) -> None:
        self.check(sympy.Le(left, right))

    def check_lt(self, left: Expr, right: Expr) -> None:
        self.check(sympy.Lt(left, right))

    # Similar to the functions guard_or_false/guard_or_true in symbolic_shapes.py
    # but operates on sympy expressions instead of symnodes. see Note [guard_or_].
    def guard_or_false(self, left):
        import torch.fx.experimental._config as exp_config

        if exp_config.backed_size_oblivious:
            static_val = self.shape_env._maybe_evaluate_static(left)
            if static_val is not None:
                return static_val
            return False
        return self.evaluate_expr(left, fallback_value=False)

    def guard_or_true(self, left):
        import torch.fx.experimental._config as exp_config

        if exp_config.backed_size_oblivious:
            static_val = self.shape_env._maybe_evaluate_static(left)
            if static_val is not None:
                return static_val
            return True
        return self.evaluate_expr(left, fallback_value=True)

    # The evaluate functions evaluate some symbolic sympy expression
    # (NB: not necessarily an Expr) and return what the concrete result
    # is, guarding on the expression being that result

    # NB: write evaluate_expr(sympy.Lt(a, b)) rather than evaluate_expr(a < b)
    # as this will ensure that you actually have a sympy'ified expression,
    # and will prevent you from incorrectly writing evaluate_expr(a == b)
    # which does the wrong thing if a or b is a sympy expression
    def evaluate_expr(
        self,
        left: Union[Expr, sympy.logic.boolalg.Boolean],
        size_oblivious: bool = False,
        fallback_value: Optional[bool] = None,
    ) -> bool:
        assert isinstance(left, (Expr, sympy.logic.boolalg.Boolean)), type(left)
        return self.shape_env.evaluate_expr(
            sympy.sympify(left),
            size_oblivious=size_oblivious,
            fallback_value=fallback_value,
        )

    def is_size_one_or_false(self, size: Expr) -> bool:
        """Return True if size equals 1.

        Unbacked symbolic sizes return False without introducing a guard.
        """
        return self.guard_or_false(sympy.Eq(size, 1))

    def evaluate_min(self, left: Expr, right: Expr) -> Expr:
        """return the smaller of left and right, and guard on that choice"""
        if isinstance(left, Expr):
            left = sympy_subs(left, self.inv_precomputed_replacements)  # type: ignore[arg-type]
        if isinstance(right, Expr):
            right = sympy_subs(right, self.inv_precomputed_replacements)  # type: ignore[arg-type]
        try:
            lv = self.size_hint_or_throw(left)
            rv = self.size_hint_or_throw(right)
        except TypeError:  # unbacked symints
            if left == right or self.statically_known_leq(left, right):
                return left
            if self.statically_known_leq(right, left):
                return right
            gcd = sympy.gcd(left, right)
            if left == gcd:  # handle `min(10*u0, u0)` etc
                return left
            if right == gcd:
                return right
            raise TypeError(
                f"evaluate_min({left}, {right}) with unbacked symints"
            ) from None
        if lv <= rv:
            self.check_leq(left, right)
            return left
        else:
            self.check_leq(right, left)
            return right

    def evaluate_max(self, left: Expr, right: Expr) -> Expr:
        """return the larger of left and right, and guard on that choice"""
        # Always choose the opposite of eval min for consistency
        # This means min(a, b) and max(a, b) produce the same guards
        min_val = self.evaluate_min(left, right)
        return right if min_val is left else left

    def guard_int(self, expr: Union[Expr, int]) -> int:
        """
        Similar to guard_int in symbolic_shapes.py, except this function works with SymPy
        expressions instead of SymNodes. It extracts the value represented by expr from shapeEnv
        and specialize the compiled graph on it. Raises an error if the result cannot be
        determined due to unhinted or unbacked symbols.
        """
        if isinstance(expr, int):
            return expr
        val = self.size_hint_or_throw(expr)
        self.check_equals(expr, sympy.Integer(val))
        return int(val)

    def guard_int_seq(self, left: Sequence[Union[Expr, int]]) -> list[int]:
        """
        Apply guard_int on a sequence of inputs.
        """
        return [self.guard_int(x) for x in left]

    def remove_precomputed_replacements(self, expr: Expr) -> Expr:
        if any(symbol_is_type(s, SymT.PRECOMPUTED_SIZE) for s in expr.free_symbols):  # type: ignore[attr-defined]
            return sympy_subs(expr, self.inv_precomputed_replacements)  # type: ignore[arg-type]
        return expr

    def symbolic_hint(
        self,
        expr: Union[Expr, int],
        # Only flip this flag if you don't plan on guarding/adding runtime
        # asserts based on this value and promise to only use this value
        # in a heuristic nature.
        use_user_provided_hint_override: bool = False,
    ) -> Union[Expr, int]:
        if isinstance(expr, int):
            return expr
        # Substitute all hints into expr, but leave unbacked symints alone
        expr = self.simplify(expr)
        if not isinstance(expr, Expr):
            assert isinstance(expr, int)
            return expr
        free_symbols = expr.free_symbols
        if not free_symbols:
            try:
                return int(expr)  # type: ignore[return-value]
            except TypeError:
                return expr  # inf/nan/I

        expr = self.remove_precomputed_replacements(expr)

        if use_user_provided_hint_override:
            expr = sympy_subs(expr, self.var_to_hint_override)

        return sympy_subs(expr, self.backed_var_to_val)

    def size_hint(
        self,
        expr: Union[Expr, int],
        *,
        fallback: Optional[int] = None,
    ) -> int:
        if isinstance(expr, SymInt):
            raise TypeError(
                "wrong API usage!, use size_hint from torch.fx.experimental.symbolic_shapes or pass sympy expressions instead"
            )

        out = self.symbolic_hint(
            expr,
            use_user_provided_hint_override=fallback is not None,
        )
        if not isinstance(out, (int, sympy.Integer)) and fallback is not None:
            # Use the provided heuristic fallback hint
            unbacked_sym_vrs = {
                s: self.shape_env.var_to_range.get(s, None) for s in out.free_symbols
            }
            if all(vr is not None for vr in unbacked_sym_vrs.values()):
                hint_vr = bound_sympy(out, unbacked_sym_vrs)  # type: ignore[arg-type]
                if isinstance(hint_vr.lower, (int, sympy.Integer)):
                    fallback = max(fallback, int(hint_vr.lower))
                if isinstance(hint_vr.upper, (int, sympy.Integer)):
                    fallback = min(fallback, int(hint_vr.upper))
            return fallback

        try:
            return int(out)
        except Exception:
            log.debug("failed on: %s", out)
            raise

    def size_hint_or_throw(self, expr: Union[Expr, int]) -> int:
        # Like size_hint but there's no fallback for unbacked symints, so it throws.
        out = self.symbolic_hint(expr)
        try:
            return int(out)
        except Exception:
            log.debug("failed on: %s", out, exc_info=True)
            raise

    def optimization_hint_with_override(
        self,
        expr: Union[Expr, int],
        hint_override: Optional[int],
    ) -> int:
        r"""Return a concrete integer hint for an expression, with optional override.
        This is used in dynamic dispatch scenarios where callers may want to
        provide a specific hint value rather than computing one from the expression.
        The resolution order is:
        1. If ``expr`` simplifies to a static integer, return that value
           (``hint_override`` is ignored for static shapes).
        2. If ``expr`` is dynamic and ``hint_override`` is not ``None``,
           return ``hint_override``.
        3. Otherwise, compute a hint via :meth:`atomically_apply_size_hint`
           with ``fallback=config.unbacked_symint_fallback``.
        Args:
            expr (Expr or int): The expression to get a hint for.
            hint_override (int, optional): If provided and ``expr`` is dynamic,
                this value is returned instead of computing a hint.
        Returns:
            int: A concrete integer hint for the expression.
        """
        simplified = self.simplify(expr)
        if isinstance(simplified, int):
            return simplified
        if isinstance(simplified, sympy.Integer):
            return int(simplified)
        # Dynamic shape: use hint_override if set, else atomically_apply_size_hint
        if hint_override is not None:
            return hint_override
        return self.atomically_apply_size_hint(
            expr, fallback=config.unbacked_symint_fallback
        )

    def optimization_hints_with_override(
        self,
        exprs: Iterable[Union[Expr, int]],
        hint_override: Optional[int],
    ) -> tuple[int, ...]:
        """
        Like optimization_hint_with_override but for a sequence of expressions.
        Returns a tuple of concrete integer hints.
        """
        return tuple(
            self.optimization_hint_with_override(e, hint_override) for e in exprs
        )

    def size_hints(
        self,
        exprs: Iterable[Union[Expr, int]],
        *,
        fallback: Optional[int] = None,
    ) -> tuple[int, ...]:
        return tuple(
            self.size_hint(
                x,
                fallback=fallback,
            )
            for x in exprs
        )

    def size_hints_or_throw(
        self,
        exprs: Iterable[Union[Expr, int]],
    ) -> tuple[int, ...]:
        # Like size_hints but there's no fallback for unbacked symints, so it throws.
        return tuple(self.size_hint_or_throw(x) for x in exprs)

    def _lru_cache(self, fn, maxsize=None):
        """
        Wrapper around functools.lru_cache that clears when replacements
        has been invalidated.
        """
        fn_cache = functools.lru_cache(maxsize)(fn)
        prior_len = len(self.replacements)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal prior_len
            if prior_len != len(self.replacements):
                prior_len = len(self.replacements)
                fn_cache.cache_clear()
            return fn_cache(*args, **kwargs)

        return wrapper

    def make_stride_vars_cache(self):
        cache = self._lru_cache(self._stride_vars)

        def stride_vars(
            index: Expr,
            vars: Sequence[sympy.Symbol],
            support_vars: Optional[Sequence[sympy.Symbol]] = None,
        ) -> list[Expr]:
            if not support_vars:
                support_vars = vars
            return cache(index, tuple(vars), tuple(support_vars))

        return stride_vars

    def _stride_vars(
        self,
        index: Expr,
        vars: Sequence[sympy.Symbol],
        support_vars: Sequence[sympy.Symbol],
    ) -> list[Expr]:
        """Convert an indexing expression back into strides

        NOTE: This is only valid if the index is a standard strided offset
        calculation. e.g. 10 * ModularIndexing(i0 + 1, 1, 2) would give a
        stride of -10 because the index wraps around after the first element

        """
        strides = []
        index = self.simplify(index)
        # remove any offset
        index = index - sympy_subs(
            index, {v: sympy.S.Zero for v in support_vars if v != 0}
        )
        for i in range(len(vars)):
            # drop all the other dims
            index_dim = sympy_subs(
                index,
                {
                    support_vars[j]: sympy.S.Zero
                    for j in range(len(support_vars))
                    if vars[i] != support_vars[j] and support_vars[j] != 0
                },
            )
            v = vars[i]
            if v == 0:
                strides.append(sympy.S.Zero)
            else:
                # TODO(jansel): should we use sympy.diff here?
                strides.append(
                    sympy_subs(index_dim, {v: sympy.S.One})
                    - sympy_subs(index_dim, {v: sympy.S.Zero})
                )
        return strides

    def _get_unbacked_replacements(self) -> dict[Expr, Expr]:
        if self.unbacked_replacements is not None:
            return self.unbacked_replacements

        class CanonicalExprFinder:
            """
            Purpose:
            A disjoint-set/union-find data structure that can return the
            "canonical" expression for a group of equivalent expressions.
            - The canonical expression must come from the input eq_graph.
            - The heuristics used to choose a leader determines which
            expression becomes the canonical expression.

            Problem:
            Given any unbacked expression, we should be able to find a size_hint
            for the unbacked expression, that adheres to the ShapeEnv's deferred
            runtime assertions. Otherwise, we may generate conflicting size hints.
            In other words, even though we know u0 + s0 == u2, we may generate
            size hints, such that, size_hint(u0 + s0) != size_hint(u2).
            NOTE: At this time, only deferred runtime asserts that are equalities
            (i.e. Eq(lhs, rhs)) are considered in this data structure.

            Examples:
            - u0 + u1 == 9000, then find_expr(u0 + u1) == find_expr(9000)
            - u0 + u1 == s9, then find_expr(u0 + u1) == find_expr(s9)
            - u0 + s0 == u10, then find_expr(u0 + s0) == find_expr(u10)

            Inputs:
            - equality_graph: An adjacency set of expressions where the edge
            connects two expressions that are found equal to each other. The
            edges are sourced from ShapeEnv's deferred_runtime_asserts.

            Usage:
            - Call union_expr(a, b) to merge a & b into a single set which
            shares the same canonical expression.
            - Call find_expr(x) to find the canonical expression for x.
            """

            def __init__(self, eq_graph: dict[Expr, OrderedSet[Expr]]):
                self.eq_graph = eq_graph
                self.expressions = list(eq_graph.keys())
                self.reverse_expressions = {
                    expr: i for i, expr in enumerate(self.expressions)
                }
                # Each node is its own leader/parent initially
                self.leader = list(range(len(self.expressions)))
                # Track rank for union-by-rank
                self.rank = [1] * len(self.expressions)

                # Takes each edge from the undirected graph and starts merging them.
                self._build_canonical_expr_mapping()

            def _build_canonical_expr_mapping(self):
                for expr, edges in self.eq_graph.items():
                    for adj in edges:
                        self.union_expr(expr, adj)

            def union_expr(self, a: Expr, b: Expr):
                return self.union(
                    self.reverse_expressions[a], self.reverse_expressions[b]
                )

            def union(self, a: int, b: int):
                rootA = self.find(a)
                rootB = self.find(b)
                if rootA == rootB:
                    return False  # already connected
                leader, other = self.choose_leader(rootA, rootB)
                self.leader[other] = leader
                self.rank[leader] += self.rank[other]
                return True

            def find_expr(self, expr: Expr):
                parent = self.find(self.reverse_expressions[expr])
                return self.expressions[parent]

            def find(self, x: int):
                # Path compression
                if self.leader[x] != x:
                    self.leader[x] = self.find(self.leader[x])
                return self.leader[x]

            def choose_leader(self, a: int, b: int):
                """
                The leader will become the canonical expression.

                Here are the heuristics used for choosing a leader:
                1. Backed expression or constants preferred over unbacked expr
                2. Simpler sub-expr when one contains the other
                3. Higher frequency across equalities from deferred runtime assertions
                4. Rank/size of the set
                5. Fallback to sympy.Basic.compare
                """

                def _choose(x: int, y: int) -> bool:
                    lhs, rhs = self.expressions[x], self.expressions[y]

                    # Prefer replacing unbacked exprs with backed expressions/constants.
                    # Examples:
                    # u0 + s3 ==> s0 + s1, then leader is s0 + s1
                    # u2 ==> 300, then leader is 300
                    any_unbacked_lhs = has_free_unbacked_symbols(lhs)
                    any_unbacked_rhs = has_free_unbacked_symbols(rhs)
                    if any_unbacked_lhs != any_unbacked_rhs:
                        return bool(any_unbacked_rhs)

                    # Handles cases where LHS contains the RHS. In other words,
                    # RHS is a sub-expression of LHS. For example:
                    # s1 * Max(2, u0) ==> Max(2, u0), then leader is Max(2, u0)
                    if lhs.has(rhs):
                        return False
                    elif rhs.has(lhs):
                        return True

                    # Prefer expressions that come up more often.
                    degrees_lhs = len(self.eq_graph[lhs])
                    degrees_rhs = len(self.eq_graph[rhs])
                    if degrees_lhs != degrees_rhs:
                        return degrees_lhs > degrees_rhs

                    # Try to apply union-by-rank optimization to flatten the
                    # leader trees.
                    if self.rank[x] != self.rank[y]:
                        return self.rank[x] > self.rank[y]

                    # Fallback to sympy.Basic.compare for a deterministic ordering.
                    return lhs.compare(rhs) == -1

                if _choose(a, b):
                    return a, b
                return b, a

        # Build an undirected graph using ShapeEnv's deferred runtime assertions.
        self.equality_graph: dict[Expr, OrderedSet[Expr]] = defaultdict(OrderedSet)
        for assertions in self.shape_env.deferred_runtime_asserts.values():
            for assertion in assertions:
                if not isinstance(assertion.expr, sympy.Equality):
                    # We're ignoring other relationals for now. If you need to
                    # account for relationals, then you may need a solver solution.
                    continue
                lhs = sympy.sympify(assertion.expr.lhs)  # sympify helps with ints
                rhs = sympy.sympify(assertion.expr.rhs)
                self.equality_graph[lhs].add(rhs)
                self.equality_graph[rhs].add(lhs)

        # Use the undirected graph to create a DSU data structure, so we can
        # query for a "canonical" expression.
        uf = CanonicalExprFinder(self.equality_graph)

        # Start building the unbacked replacements mapping using CanonicalExprFinder
        # The mapping is from Expr to its "canonical" Expr.
        self.unbacked_replacements = {}
        for expr in self.equality_graph:
            canonical_expr = uf.find_expr(expr)
            if expr != canonical_expr:
                self.unbacked_replacements[expr] = canonical_expr

        return self.unbacked_replacements

    @functools.lru_cache  # noqa: B019
    def _sub_unbacked_exprs(self, expr: Expr) -> Expr:
        # it's fine to cache this fn since self is a singleton
        replacements = self._get_unbacked_replacements()

        # consider making this threshold configurable
        sub_cnt_limit = 30
        sub_cnt = 0
        while sub_cnt < sub_cnt_limit:
            new_expr = expr.subs(replacements)
            if new_expr == expr:
                return new_expr
            expr = sympy.factor(new_expr)
            sub_cnt += 1

        log.warning("Substitution limit (%d) reached w/ %s", sub_cnt_limit, expr)
        return expr

    def atomically_apply_size_hint(
        self,
        expr: Union[Expr, int],
        *,
        fallback: Optional[int] = None,
    ) -> Union[Expr, int]:
        if isinstance(expr, (int, sympy.Integer)):
            return int(expr)

        if has_free_unbacked_symbols(expr):
            # Make sure to substitute with the factored version
            # e.g. 10*(s0 + u0) instead of 10*s0 + 10*u0
            expr = self._sub_unbacked_exprs(sympy.factor(expr))

        # For multiple expressions that depend on an unbacked symint,
        # we want to compute them consistently for a size hint we have chosen.
        # So, recursively compute expressions via size hints of contained symbols.
        # For example: u1 * u2 - 10 ==> fallback * fallback - 10
        assert isinstance(expr, Expr), type(expr)
        free_symbols = expr.free_symbols
        size_dict = {
            symbol: V.graph.sizevars.size_hint(symbol, fallback=fallback)
            for symbol in free_symbols
        }
        return expr.subs(size_dict)

    def offset_var(self, index: Expr, vars: Sequence[sympy.Symbol]) -> Expr:
        """Extract offset part of an indexing expression"""
        index = self.simplify(index)
        return sympy_subs(index, {v: sympy.S.Zero for v in vars if v != 0})

    def stride_hints(
        self,
        index: Expr,
        vars: Sequence[sympy.Symbol],
        support_vars: Optional[Sequence[sympy.Symbol]] = None,
    ) -> list[int]:
        for v in index.free_symbols:
            if symbol_is_type(v, SymT.INDIRECT):  # type: ignore[attr-defined]
                index = sympy_subs(index, {v: 0})  # type: ignore[dict-item]
        result = []
        for s in self.stride_vars(index, vars, support_vars):
            try:
                result.append(self.size_hint_or_throw(s))
            except TypeError:
                result.append(0)
        return result

    def stride_order(self, index: Expr, vars: list[sympy.Symbol]) -> list[int]:
        strides = tuple(map(abs, self.stride_hints(index, vars)))
        order = list(range(len(strides)))
        order.sort(key=lambda x: (strides[x] == 0, strides[x]))
        return order

    def lookup_precomputed_size(self, expr: Expr) -> Expr:
        if (
            isinstance(expr, (int, sympy.Symbol, sympy.Number))
            or expr.is_number
            or expr.is_symbol
        ):
            return expr
        expr = self.remove_precomputed_replacements(expr)
        if expr not in self.precomputed_replacements:
            sym = sympy_index_symbol_with_prefix(
                SymT.PRECOMPUTED_SIZE, len(self.precomputed_replacements)
            )
            self.precomputed_replacements[expr] = sym
            self.inv_precomputed_replacements[sym] = expr
        return self.precomputed_replacements[expr]

    def free_symbols(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet(self.backed_var_to_val.keys()) - OrderedSet(
            self.replacements.keys()
        )

    def combine_modular_indexing_pairs(self, index: sympy.Expr) -> sympy.Expr:
        """
        A pair of special ModularIndexing can be combined.

        E.g. ModularIndexing(ModularIndexing(x, 1, a), 1, b)
        We can simplify this to ModuleIndexing(x, 1, b), if
        1. x is non negative integer
        2. a and b are positive integers
        3. a is a multiple of b.
        """

        def _check_args(x, div, mod, is_first):
            if not isinstance(div, sympy.Integer) or not isinstance(mod, sympy.Integer):
                return False
            if div != 1:
                return False
            if mod <= 0:
                return False

            if is_first:
                # first ModularIndexing should contains a nested ModularIndex
                if not isinstance(x, ModularIndexing):
                    return False
            else:
                # second ModularIndexing should contains a non-negative
                # symbol
                if not isinstance(x, sympy.Symbol) or not self.statically_known_geq(
                    x, 0
                ):
                    return False
            return True

        if isinstance(index, ModularIndexing):
            x, div, mod = index.args

            if not _check_args(x, div, mod, True):
                return index

            x2, div2, mod2 = x.args

            if not _check_args(x2, div2, mod2, False):
                return index

            if mod2 % mod != 0:
                return index

            return ModularIndexing(x2, 1, mod)

        return index

    def expand_floor_div(
        self, index: sympy.Expr
    ) -> Union[bool, tuple[sympy.Expr, sympy.Expr]]:
        """
        Expand the FloorDiv to the entire expression so that the expression may
        be simplified.

        E.g., for a 2D contiguous tensor with shape [a, 2 * b], and index variables
        x1, x2, index expression 'x1 * 2b + x2' can be easily combined.
        But index expression 'x1 * b + x2 // 2' can not.
        By expanding the FloorDiv to the entire expression, we get
        '(x1 * 2b + x2) // 2'. This transformation allows us to merge loops
        for the numerator!

        Return false if this optimization can be applied;
        Return the new expression and the denominator otherwise.
        The original expression will be equivalent to 'new_expression // denominator'
        """
        if not isinstance(index, sympy.Add):
            return False
        terms = index.args

        if len(terms) < 2:
            return False
        floor_div_index = -1
        varlist = []
        factorlist = []
        for idx, term in enumerate(terms):
            if isinstance(term, sympy.Mul):
                # For dynamic shape, term like '2*s1*x1' has 3 child nodes.
                # - A integer for 2
                # - A symbol for s1
                # - A symbol for x1
                # Skip for now.
                if len(term.args) != 2:
                    return False
                factor, var = term.args
                varlist.append(var)
                factorlist.append(factor)
                if not isinstance(factor, sympy.Integer) or not isinstance(
                    var, sympy.Symbol
                ):
                    return False
                # It's easier to reason about the correceness of the transformation
                # for non-negative integers.
                if not self.statically_known_geq(var, 0):
                    return False
            elif isinstance(term, FloorDiv):
                var, factor = term.args
                if not isinstance(factor, sympy.Integer) or not isinstance(
                    var, sympy.Symbol
                ):
                    return False
                if not self.statically_known_geq(var, 0):
                    return False
                if floor_div_index >= 0:
                    # can not handle multi FloorDiv yet
                    return False

                floor_div_index = idx
                varlist.append(var)
                # this factor is denominator
                factorlist.append(factor)
            else:
                return False

        if floor_div_index < 0:
            return False

        # Construct the new expression and remember the denominator
        denominator = factorlist[floor_div_index]
        new_index = sympy.S.Zero

        for var, factor, idx in zip(varlist, factorlist, itertools.count()):
            if idx == floor_div_index:
                new_index += var
            else:
                new_index += (factor * denominator) * var

        return new_index, denominator


def join_dimensions(expr: Expr) -> Expr:
    if not isinstance(expr, sympy.Add) or not expr.has(ModularIndexing):
        return expr  # fast exit path
    return _join_dimensions_cached(expr)


@functools.lru_cache(256)
def _join_dimensions_cached(expr: Expr) -> Expr:
    """
    ModularIndexing(i0, 1, 32) + 32 * ModularIndexing(i0, 32, 4)
    becomes
    ModularIndexing(i0, 1, 128)
    ModularIndexing(i0, 1, 32) + 32 * FloorDiv(i0, 32)
    becomes i0


    This type of pattern can come from view operations
    """
    assert isinstance(expr, sympy.Add)

    scale = sympy.Wild("scale", exclude=[0], integer=True)
    base = sympy.Wild("base", integer=True)
    divisor = sympy.Wild("divisor", integer=True)
    mod1 = sympy.Wild("modulus", integer=True)
    mod2 = sympy.Wild("modulus2", integer=True)
    for term1 in expr.args:
        m1 = term1.match(scale * ModularIndexing(base, divisor, mod1))
        if m1:
            for term2 in expr.args:
                m2 = term2.match(
                    m1[scale]
                    * m1[mod1]
                    * ModularIndexing(m1[base], m1[divisor] * m1[mod1], mod2)
                )
                if m2 and term1 != term2:
                    expr = join_dimensions(
                        expr
                        - term1
                        - term2
                        + m1[scale]
                        * ModularIndexing(m1[base], m1[divisor], m1[mod1] * m2[mod2])
                    )
                    return expr
    for term1 in expr.args:
        m1 = term1.match(scale * ModularIndexing(base, divisor, mod1))
        if m1:
            for term2 in expr.args:
                m2 = term2.match(
                    m1[scale] * m1[mod1] * FloorDiv(m1[base], m1[divisor] * m1[mod1])
                )
                if m2 is not None:  # in case of success we get an empty dict here
                    expr = join_dimensions(
                        expr
                        - term1
                        - term2
                        + m1[scale] * FloorDiv(m1[base], m1[divisor])
                    )
                    return expr
    return expr


class SimplifyIndexing(V.WrapperHandler):  # type: ignore[name-defined]
    """
    A wrapper around .virtualize.ops that uses var range information to
    simplify ModularIndexing/FloorDiv.
    """

    def __init__(self, inner, var_ranges: VarRanges) -> None:
        super().__init__(inner)
        self.name = "SimplifyIndexing"
        self._simplify: Callable[[Expr], Expr] = (
            lambda index: V.graph.sizevars.simplify_with_ranges(index, var_ranges)
        )

    def load(self, name: str, index: sympy.Expr):
        return self._inner.load(name, self._simplify(index))

    def store(self, name, index, value, mode=None):
        return self._inner.store(name, self._simplify(index), value, mode=mode)

    def store_reduction(self, name, index, value):
        return self._inner.store_reduction(name, self._simplify(index), value)

    def index_expr(self, index, dtype):
        return self._inner.index_expr(self._simplify(index), dtype)

    def check_bounds(self, index, size, lower, upper):
        return self._inner.check_bounds(self._simplify(index), size, lower, upper)
