# mypy: allow-untyped-defs
import functools
import itertools
import logging
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import sympy
from sympy import Expr

from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, ShapeEnv
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.symbol import symbol_is_type, SymT
from torch.utils._sympy.value_ranges import bound_sympy, IntInfinity, ValueRanges

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


def evaluate_expr(
    shape_env: ShapeEnv,
    expr: Union[sympy.Basic, bool],
    axioms: Optional[Tuple[sympy.Expr]] = None,
    var_to_range: Optional[Tuple[Tuple[sympy.Symbol, ValueRanges[Any]]]] = None,
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
    def __init__(self, shape_env=None) -> None:
        super().__init__()
        if shape_env is None:
            shape_env = ShapeEnv()
        self.shape_env = shape_env
        self.var_to_val = self.shape_env.var_to_val
        self.replacements: Dict[sympy.Symbol, Expr] = self.shape_env.replacements
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
        self.precomputed_replacements: Dict[Expr, sympy.Symbol] = {}
        self.inv_precomputed_replacements: Dict[sympy.Symbol, Expr] = {}
        self.stride_vars = self.make_stride_vars_cache()
        self.simplify_with_ranges = self.make_simplify_with_ranges_cache()
        self._simplify_loops = self.make_simplify_loops_cache()

    def simplify(self, expr: Expr):
        return sympy.expand(expr).xreplace(self.replacements)

    def make_simplify_with_ranges_cache(self) -> Callable[[Expr, VarRanges], Expr]:
        """
        self._simplify_with_ranges() can be expensive, cache its results
        """
        cache: Dict[Tuple[Any, ...], Expr] = {}
        replacement_count = len(self.replacements)

        def simplify_with_ranges(expr: Expr, var_ranges: VarRanges) -> Expr:
            nonlocal replacement_count
            if replacement_count != len(self.replacements):
                # new replacements invalidates cached results
                cache.clear()
                replacement_count = len(self.replacements)
            key = (expr, *var_ranges.items())
            result = cache.get(key, None)
            if result is None:
                result = self._simplify_with_ranges(expr, var_ranges)
                cache[key] = result
            return result

        return simplify_with_ranges

    def make_simplify_loops_cache(self):
        """
        self._simplify_with_ranges() can be expensive, cache its results
        """
        cache: Dict[Tuple[Any, ...], Any] = {}
        replacement_count = len(self.replacements)

        def simplify_loops(index_vars, sizes, index_formulas):
            nonlocal replacement_count
            if replacement_count != len(self.replacements):
                # new replacements invalidates cached results
                cache.clear()
                replacement_count = len(self.replacements)
            key = (*index_vars, *sizes, *index_formulas)
            result = cache.get(key, None)
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
            Tuple[Tuple[sympy.Symbol, ValueRanges[sympy.Expr]]],
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
        self, index_vars: List[sympy.Symbol], sizes, index_formulas
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
    #
    # The statically_known_* family of functions below replaces a prior system, called maybe_guard_*. The prior system
    # operated by providing essentially a question, where the size hinted values were evaluated. If the condition was
    # true, we add a guard and return True, otherwise, False.
    #
    # def maybe_guard_foo(args):
    #   if size_hinted_check(args):
    #       return False # No guard, no optim
    #   guard(args) # Make a guard
    #   return True # Safe to apply optimization
    #
    # The prior system incurred a guard, and green lit an optimization.
    #
    # The new system works in reverse - in the new system, if we know that the inputs are static, and evaluate the
    # condition as true, we green light the optimization, and we do not incur a guard. If we cannot prove that, we
    # return False.
    #
    # def maybe_guard_foo(args):
    #   if all_static(args):
    #       return True # Safe to apply optimization
    #   else:
    #       return False # No guard, no optim

    # See Note - [On Statically Known]

    def is_expr_static_and_true(self, expr: Union[sympy.Basic, bool]) -> bool:
        return evaluate_expr(self.shape_env, expr)

    def statically_known_equals(
        self, left: Union[Expr, int], right: Union[Expr, int]
    ) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left and right are equal.
        """
        return self.is_expr_static_and_true(sympy.Eq(left, right))  # type: ignore[arg-type]

    # See Note - [On Statically Known]
    def statically_known_list_equals(self, left: List[Expr], right: List[Expr]) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left and right lists are equal.
        """
        return len(left) == len(right) and all(
            self.statically_known_equals(l, r) for l, r in zip(left, right)
        )

    # See Note - [On Statically Known]
    def statically_known_leq(self, left: Expr, right: Union[Expr, int]) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is less than or equal to right.
        """
        expr = left <= right
        return self.is_expr_static_and_true(expr)

    # See Note - [On Statically Known]
    def statically_known_geq(self, left: Expr, right: Union[Expr, int]) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is greater than or equal to right.
        """
        expr = left >= right
        return self.is_expr_static_and_true(expr)

    # See Note - [On Statically Known]
    def statically_known_lt(self, left: Expr, right: Union[Expr, int]) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is less than right.
        """
        expr = left < right
        return self.is_expr_static_and_true(expr)

    # See Note - [On Statically Known]
    def statically_known_gt(self, left: Expr, right: Union[Expr, int]) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is greater than right.
        """
        expr = left > right
        return self.is_expr_static_and_true(expr)

    # See Note - [On Statically Known]
    def statically_known_multiple_of(
        self, numerator: Expr, denominator: Union[Expr, int]
    ) -> bool:
        """
        Return a bool indicating if it is sound to optimize for the numerator being a multiple of the denominator.
        """
        if free_unbacked_symbols(numerator) or free_unbacked_symbols(denominator):
            return False
        expr = sympy.Eq(numerator % denominator, 0)
        return self.is_expr_static_and_true(expr)  # type: ignore[arg-type]

    # See Note - [On Statically Known]
    def statically_known_power_of_2(self, expr: Expr) -> bool:
        """
        Returns a bool indicating if x is known to be a power of 2.
        """
        return isinstance(expr, sympy.Integer) and is_power_of_2(int(expr))

    # The guard functions require you to ALREADY KNOW that a particular
    # condition holds.  If you don't know (you want to guard on an expression
    # being a particular value, and then get access to that value), use
    # the evaluate functions.

    def guard_equals(self, left: Expr, right: Expr) -> Expr:
        if isinstance(left, Expr):
            left = sympy_subs(left, self.inv_precomputed_replacements)  # type: ignore[arg-type]
        if isinstance(right, Expr):
            right = sympy_subs(right, self.inv_precomputed_replacements)  # type: ignore[arg-type]

        expr = sympy.Eq(left, right)
        static_expr = self.shape_env._maybe_evaluate_static(expr)

        if static_expr is not None:
            assert bool(static_expr)
            return left

        assert self.shape_env.defer_runtime_assert(expr, "guard_equals")
        return left

    def guard_leq(self, left: Expr, right: Expr) -> None:
        return self.guard_lt(left, right + 1)

    def guard_lt(self, left: Expr, right: Expr) -> None:
        expr = sympy.Lt(left, right)
        static_expr = self.shape_env._maybe_evaluate_static(expr)

        if static_expr is not None:
            assert bool(static_expr)
            return

        assert self.shape_env.defer_runtime_assert(expr, "guard_lt")

    def guarded_order(self, seq):
        """
        Return the order of a sequence as a permutation of range(len(seq)) and guard on that order not changing.
        """
        seq = [*map(self.remove_precomputed_replacements, seq)]
        seq = [(self.size_hint(var), orig_idx, var) for orig_idx, var in enumerate(seq)]
        seq.sort()
        order = [-1] * len(seq)
        last_var = None
        for new_index, (_, orig_index, var) in enumerate(seq):
            order[orig_index] = new_index
            if last_var is not None:
                self.guard_leq(last_var, var)
            last_var = var
        return order

    # The evaluate functions evaluate some symbolic sympy expression
    # (NB: not necessarily an Expr) and return what the concrete result
    # is, guarding on the expression being that result

    # NB: write evaluate_expr(sympy.Lt(a, b)) rather than evaluate_expr(a < b)
    # as this will ensure that you actually have a sympy'ified expression,
    # and will prevent you from incorrectly writing evaluate_expr(a == b)
    # which does the wrong thing if a or b is a sympy expression
    def evaluate_expr(self, left: Union[Expr, sympy.logic.boolalg.Boolean]) -> bool:
        assert isinstance(left, (Expr, sympy.logic.boolalg.Boolean)), type(left)
        return self.shape_env.evaluate_expr(sympy.sympify(left))

    def evaluate_min(self, left: Expr, right: Expr) -> Expr:
        """return the smaller of left and right, and guard on that choice"""
        if isinstance(left, Expr):
            left = sympy_subs(left, self.inv_precomputed_replacements)  # type: ignore[arg-type]
        if isinstance(right, Expr):
            right = sympy_subs(right, self.inv_precomputed_replacements)  # type: ignore[arg-type]
        try:
            lv = self.size_hint(left)
            rv = self.size_hint(right)
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
            self.guard_leq(left, right)
            return left
        else:
            self.guard_leq(right, left)
            return right

    def evaluate_max(self, left: Expr, right: Expr) -> Expr:
        """return the larger of left and right, and guard on that choice"""
        # Always choose the opposite of eval min for consistency
        # This means min(a, b) and max(a, b) produce the same guards
        min_val = self.evaluate_min(left, right)
        return right if min_val is left else left

    def evaluate_static_shape(self, left: Union[Expr, int]) -> int:
        if isinstance(left, int):
            return left
        right = self.size_hint(left)
        self.guard_equals(left, sympy.Integer(right))
        return int(right)

    def evaluate_static_shapes(self, left: Sequence[Union[Expr, int]]) -> List[int]:
        return [self.evaluate_static_shape(x) for x in left]

    def remove_precomputed_replacements(self, expr: Expr) -> Expr:
        if any(symbol_is_type(s, SymT.PRECOMPUTED_SIZE) for s in expr.free_symbols):  # type: ignore[attr-defined]
            return sympy_subs(expr, self.inv_precomputed_replacements)  # type: ignore[arg-type]
        return expr

    def symbolic_hint(self, expr: Union[Expr, int]) -> Union[Expr, int]:
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
        return sympy_subs(expr, self.var_to_val)

    def size_hint(
        self, expr: Union[Expr, int], *, fallback: Optional[int] = None
    ) -> int:
        out = self.symbolic_hint(expr)
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

    def size_hints(
        self,
        exprs: Iterable[Expr],
        *,
        fallback: Optional[int] = None,
    ) -> Tuple[int, ...]:
        return tuple(self.size_hint(x, fallback=fallback) for x in exprs)

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
        ) -> List[Expr]:
            if not support_vars:
                support_vars = vars
            return cache(index, tuple(vars), tuple(support_vars))

        return stride_vars

    def _stride_vars(
        self,
        index: Expr,
        vars: Sequence[sympy.Symbol],
        support_vars: Sequence[sympy.Symbol],
    ) -> List[Expr]:
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

    def atomically_apply_size_hint(
        self, expr: Union[Expr, int], *, fallback: Optional[int] = None
    ) -> Union[Expr, int]:
        if isinstance(expr, int):
            return int(expr)

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

    def offset_var(self, index: Expr, vars: List[sympy.Symbol]) -> Expr:
        """Extract offset part of an indexing expression"""
        index = self.simplify(index)
        return sympy_subs(index, {v: sympy.S.Zero for v in vars if v != 0})

    def stride_hints(
        self,
        index: Expr,
        vars: Sequence[sympy.Symbol],
        support_vars: Optional[Sequence[sympy.Symbol]] = None,
    ) -> List[int]:
        for v in index.free_symbols:
            if symbol_is_type(v, SymT.INDIRECT):  # type: ignore[attr-defined]
                index = sympy_subs(index, {v: 0})  # type: ignore[dict-item]
        result = []
        for s in self.stride_vars(index, vars, support_vars):
            try:
                result.append(self.size_hint(s))
            except TypeError:
                result.append(0)
        return result

    def stride_order(self, index: Expr, vars: List[sympy.Symbol]) -> List[int]:
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
        return OrderedSet(self.var_to_val.keys()) - OrderedSet(self.replacements.keys())

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
                # first ModularIndexing should conatins a nested ModularIndex
                if not isinstance(x, ModularIndexing):
                    return False
            else:
                # second ModularIndexing should constains a non-negative
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
    ) -> Union[bool, Tuple[sympy.Expr, sympy.Expr]]:
        """
        Expand the FloorDiv to the entire expression so that the expression may
        be simplfied.

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
        self._simplify: Callable[
            [Expr], Expr
        ] = lambda index: V.graph.sizevars.simplify_with_ranges(index, var_ranges)

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
