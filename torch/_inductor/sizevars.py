import dataclasses
import functools
import itertools
import logging
from typing import Callable, Dict, List, Tuple

import sympy
from sympy import Expr

from torch.fx.experimental.symbolic_shapes import ShapeEnv

from . import ir
from .codegen.common import IndentedBuffer
from .utils import sympy_subs, sympy_symbol, VarRanges
from .virtualized import V

log = logging.getLogger(__name__)


@dataclasses.dataclass
class ZeroGuard:
    """
    An expression we should check equals zero.
    Guards are currently not checked.  Plan to add this later.
    """

    expr: Expr


@dataclasses.dataclass
class PositiveGuard:
    """
    An expression we should check for > 0
    Guards are currently not checked.  Plan to add this later.
    """

    expr: Expr


class SizeVarAllocator(object):
    def __init__(self, shape_env=None):
        super().__init__()
        if shape_env is None:
            shape_env = ShapeEnv()
        self.shape_env = shape_env
        self.var_to_val = self.shape_env.var_to_val
        self.guards = []
        self.replacements: Dict[sympy.Symbol, Expr] = self.shape_env.replacements
        self.need_seed = False
        self.stride_vars = self.make_stride_vars_cache()
        self.simplify_with_ranges = self.make_simplify_with_ranges_cache()
        self._simplify_loops = self.make_simplify_loops_cache()

    def seed(self):
        """
        Seed is a special variable used to hold the rng seed for a graph.

        Note this is only used by the CPU backend, we put seeds in a
        1-element tensor for the CUDA backend.
        """
        self.need_seed = True
        return sympy_symbol("seed")

    def simplify(self, expr: Expr):
        return sympy.expand(expr).xreplace(self.replacements)

    def make_simplify_with_ranges_cache(self):
        """
        self._simplify_with_ranges() can be expensive, cache its results
        """
        cache = dict()
        replacement_count = len(self.replacements)

        def simplify_with_ranges(expr: Expr, var_ranges: VarRanges):
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
        cache = dict()
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

    def _simplify_with_ranges(self, expr: Expr, var_ranges: VarRanges):
        """
        Simplify indexing expression with knowledge of the ranges of
        iteration variables.
        """
        from .ir import IndexingDiv, ModularIndexing

        expr = join_dimensions(self.simplify(expr))
        original_expr = expr

        def remove_zero_terms(base, divisor):
            """Symbols smaller than the divisor are zero"""
            for v in base.free_symbols:
                if v in var_ranges:
                    # var smaller than divisor can be removed
                    # if the rest is guaranteed to be multiple of divisor
                    rest = sympy.Wild("_rest", exclude=[v])
                    m = base.match(v + rest)
                    if m and v not in m[rest].free_symbols:
                        gcd = sympy.gcd(m[rest], divisor)
                        if gcd == divisor:
                            if self.maybe_guard_leq(var_ranges[v], divisor):
                                base = m[rest]
            return base

        def visit_indexing_div(base, divisor):
            return IndexingDiv(remove_zero_terms(base, divisor), divisor)

        def visit_modular_indexing(base, divisor, modulus):
            base = remove_zero_terms(base, divisor)
            if isinstance(base, ModularIndexing):
                # for modular indexing, biggest values from the ranges don't necessarily result in
                # the biggest result, the biggest result is modulus - 1
                base_s = base.args[2] - 1
            elif not base.has(ModularIndexing):
                # actual iteration range is to size-1
                iter_ranges = {k: v - 1 for k, v in var_ranges.items()}
                base_s = sympy_subs(base, iter_ranges)
            else:
                base_s = base
            if self.maybe_guard_lt(base_s, modulus * divisor):
                return IndexingDiv(base, divisor)
            return ModularIndexing(base, divisor, modulus)

        if expr.has(ModularIndexing):
            expr = expr.replace(
                ModularIndexing(
                    sympy.Wild("base"),
                    sympy.Wild("divisor"),
                    sympy.Wild("modulus"),
                ),
                visit_modular_indexing,
            )

        if expr.has(IndexingDiv):
            expr = expr.replace(
                IndexingDiv(
                    sympy.Wild("base"),
                    sympy.Wild("divisor"),
                ),
                visit_indexing_div,
            )

        if expr != original_expr:
            return self._simplify_with_ranges(expr, var_ranges)
        return expr

    def _simplify_loops_impl(self, index_vars, sizes, index_formulas):
        """
        Try to remove as many axis from loop iterations as possible, by:
            1) removing size==1 dimensions
            2) fuse contiguous dimensions into a single loop
            If channel_last = True, we will prevent the last dim fused with other dims
        """
        sizes = list(map(self.simplify, sizes))

        strides = [self.stride_vars(x, index_vars) for x in index_formulas]
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
                    v = sympy_symbol("_merge_tester")
                    expr1 = sympy_subs(index_formulas[k], {va: v * sizes[a], vb: 0})
                    expr2 = sympy_subs(index_formulas[k], {va: 0, vb: v})
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
                    new_index.append(sympy.Integer(0))
                else:
                    new_index.append(it.pop())
            assert not it
            return new_index

        def prune(index):
            assert len(index) == len(sizes)
            return [i for i, s in zip(index, sizes) if s is not None]

        return [x for x in sizes if x is not None], reindex, prune

    def guard_equals(self, left: Expr, right: Expr) -> Expr:
        left = sympy.expand(left)
        right = sympy.expand(right)
        if left == right:
            return left
        expr = self.simplify(left - right)
        assert self.size_hint(expr) == 0, (expr, self.size_hint(expr))
        free = list(expr.free_symbols)
        if len(free) == 0:
            assert expr == 0
            return left
        elif len(free) in (1, 2, 3):
            # remove the largest of the guarded variables
            free.sort(key=self.size_hint)
            try:
                solutions = sympy.solve(expr, free[-1])
                if (
                    len(solutions) == 1
                    and solutions[0]
                    and "/" not in str(solutions[0])
                ):
                    self.replacements[free[-1]] = solutions[0]
            except NotImplementedError:
                pass

        self.guards.append(ZeroGuard(expr))

        if len(right.free_symbols) < len(left.free_symbols):
            return right
        else:
            return left

    def maybe_guard_equals(self, left: Expr, right: Expr) -> bool:
        """if left==right, guard on that fact and return true"""
        if left == right:
            return True
        if self.size_hint(left - right) == 0:
            self.guard_equals(left, right)
            return True
        return False

    def maybe_guard_list_equals(self, left: List[Expr], right: List[Expr]) -> bool:
        """if left==right, guard on that fact and return true"""
        if len(left) != len(right):
            return False
        if all(self.size_hint(a - b) == 0 for a, b in zip(left, right)):
            for a, b in zip(left, right):
                self.guard_equals(a, b)
            return True
        return False

    def maybe_guard_leq(self, left: Expr, right: Expr) -> bool:
        try:
            if self.size_hint(left) > self.size_hint(right):
                return False
        except TypeError:
            return False
        self.guard_leq(left, right)
        return True

    def maybe_guard_lt(self, left: Expr, right: Expr) -> bool:
        try:
            if self.size_hint(left) >= self.size_hint(right):
                return False
        except TypeError:
            return False
        self.guard_lt(left, right)
        return True

    def guard_leq(self, left: Expr, right: Expr) -> None:
        return self.guard_lt(left, right + 1)

    def guard_lt(self, left: Expr, right: Expr) -> None:
        expr = self.simplify(right - left)
        assert self.size_hint(expr) > 0
        if len(expr.free_symbols) == 0:
            return
        if "-" in str(expr):
            # all vars are positive, so needs a minus sign to get negative values
            self.guards.append(PositiveGuard(expr))

    def guard_min(self, left: Expr, right: Expr) -> Expr:
        """return the smaller of left and right, and guard on that choice"""
        lv = self.size_hint(left)
        rv = self.size_hint(right)
        if lv == rv:
            return self.guard_equals(left, right)
        elif lv < rv:
            self.guard_lt(left, right)
            return left
        else:
            self.guard_lt(right, left)
            return right

    def guard_max(self, left: Expr, right: Expr) -> Expr:
        """return the larger of left and right, and guard on that choice"""
        return -self.guard_min(-left, -right)

    def maybe_guard_multiple_of(self, numerator: Expr, denominator: Expr) -> bool:
        """if denominator divides numerator, return True and guard on that fact"""
        if sympy.gcd(numerator, denominator) == denominator:
            # can prove it symbolically
            return True
        if self.size_hint(numerator) % self.size_hint(denominator) == 0:
            multiple = self.size_hint(numerator) // self.size_hint(denominator)
            self.guard_equals(multiple * denominator, numerator)
            return True
        return False

    def guard_static_shape(self, left: Expr) -> int:
        right = self.size_hint(left)
        self.guard_equals(left, sympy.Integer(right))
        return int(right)

    def __getitem__(self, val: int) -> Expr:
        return self.shape_env.create_symbol(val)

    def size_hint(self, expr: Expr) -> int:
        out = sympy_subs(sympy.expand(expr), self.var_to_val)
        return int(out)

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

        def stride_vars(index: Expr, vars: List[sympy.Symbol]) -> List[Expr]:
            return cache(index, tuple(vars))

        return stride_vars

    def _stride_vars(self, index: Expr, vars: List[sympy.Symbol]) -> List[Expr]:
        """Convert an indexing expression back into strides"""
        strides = []
        index = self.simplify(index)
        # remove any offset
        index = index - sympy_subs(index, {v: sympy.Integer(0) for v in vars if v != 0})
        for i in range(len(vars)):
            # drop all the other dims
            index_dim = sympy_subs(
                index,
                {
                    vars[j]: sympy.Integer(0)
                    for j in range(len(vars))
                    if i != j and vars[j] != 0
                },
            )
            v = vars[i]
            if v == 0:
                strides.append(sympy.Integer(0))
            else:
                # TODO(jansel): should we use sympy.diff here?
                strides.append(
                    sympy_subs(index_dim, {v: sympy.Integer(1)})
                    - sympy_subs(index_dim, {v: sympy.Integer(0)})
                )
        return strides

    def offset_var(self, index: Expr, vars: List[sympy.Symbol]) -> Expr:
        """Extract offset part of an indexing expression"""
        index = self.simplify(index)
        return sympy_subs(index, {v: sympy.Integer(0) for v in vars if v != 0})

    def stride_hints(self, index: Expr, vars: List[sympy.Symbol]) -> List[int]:
        for v in index.free_symbols:
            if v.name.startswith("indirect"):
                index = sympy_subs(index, {v: 0})
        result = []
        for s in self.stride_vars(index, vars):
            try:
                result.append(self.size_hint(s))
            except TypeError:
                result.append(0)
        return result

    def stride_order(self, index: Expr, vars: List[sympy.Symbol]) -> List[int]:
        strides = tuple(
            map(lambda x: abs(x), self.stride_hints(index, vars))
        )  # lambda to placate mypy
        order = list(range(len(strides)))
        order.sort(key=lambda x: (strides[x] == 0, strides[x]))
        return order

    def codegen(self, code: IndentedBuffer, graph_inputs: Dict[str, ir.Buffer]):
        """Assign all symbolic shapes to locals"""
        if self.need_seed:
            code.writeline(
                "seed = torch.randint(2**31, size=(), dtype=torch.int32).item()"
            )

        @functools.lru_cache(None)
        def sizeof(name):
            code.writeline(f"{name}_size = {name}.size()")
            return f"{name}_size"

        @functools.lru_cache(None)
        def strideof(name):
            code.writeline(f"{name}_stride = {name}.stride()")
            return f"{name}_stride"

        # Assign all symbolic shapes needed to local variables
        needed = set(self.var_to_val.keys()) - set(self.replacements.keys())
        added = set()

        for name, value in graph_inputs.items():
            shapes = value.get_size()
            for dim, shape in enumerate(shapes):
                shape = self.simplify(shape)
                if shape in needed:
                    needed.remove(shape)
                    added.add(shape)
                    code.writeline(f"{shape} = {sizeof(name)}[{dim}]")
                elif isinstance(shape, sympy.Symbol):
                    assert shape in added, f"{shape} is needed but not added"

        for name, value in graph_inputs.items():
            shapes = value.get_stride()
            for dim, shape in enumerate(shapes):
                shape = self.simplify(shape)
                if shape in needed:
                    needed.remove(shape)
                    code.writeline(f"{shape} = {strideof(name)}[{dim}]")
                elif isinstance(shape, sympy.Symbol):
                    assert shape in added, f"{shape} is needed but not added"
        assert not needed

    def codegen_sizevar(self, x: Expr) -> str:
        from .codegen.wrapper import pexpr

        return pexpr(self.simplify(x))

    def codegen_shape_tuple(self, shape: Tuple[Expr, ...]) -> str:
        parts = list(map(self.codegen_sizevar, shape))
        if len(parts) == 0:
            return "()"
        if len(parts) == 1:
            return f"({parts[0]}, )"
        return f"({', '.join(parts)})"


def join_dimensions(expr: Expr) -> Expr:
    from .ir import ModularIndexing

    if not isinstance(expr, sympy.Add) or not expr.has(ModularIndexing):
        return expr  # fast exit path
    return _join_dimensions_cached(expr)


@functools.lru_cache(256)
def _join_dimensions_cached(expr: Expr) -> Expr:
    """
    ModularIndexing(i0, 1, 32) + 32 * ModularIndexing(i0, 32, 4)
    becomes
    ModularIndexing(i0, 1, 128)
    ModularIndexing(i0, 1, 32) + 32 * IndexingDiv(i0, 32)
    becomes i0


    This type of pattern can come from view operations
    """
    from .ir import IndexingDiv, ModularIndexing

    assert isinstance(expr, sympy.Add)

    scale = sympy.Wild("scale", exclude=[0])
    base = sympy.Wild("base")
    divisor = sympy.Wild("divisor")
    mod1 = sympy.Wild("modulus")
    mod2 = sympy.Wild("modulus2")
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
                    m1[scale] * m1[mod1] * IndexingDiv(m1[base], m1[divisor] * m1[mod1])
                )
                if m2 is not None:  # in case of success we get an empty dict here
                    expr = join_dimensions(
                        expr
                        - term1
                        - term2
                        + m1[scale] * IndexingDiv(m1[base], m1[divisor])
                    )
                    return expr
    return expr


class SimplifyIndexing(V.WrapperHandler):  # type: ignore[name-defined]
    """
    A wrapper around .virtualize.ops that uses var range information to
    simplify ir.ModularIndexing/ir.IndexingDiv.
    """

    def __init__(self, inner, var_ranges: VarRanges):
        super().__init__(inner)
        self._simplify: Callable[
            [Expr], Expr
        ] = lambda index: V.graph.sizevars.simplify_with_ranges(index, var_ranges)

    def load(self, name: str, index: sympy.Expr):
        return self._inner.load(name, self._simplify(index))

    def store(self, name, index, value, mode=None):
        return self._inner.store(name, self._simplify(index), value, mode=mode)

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        return self._inner.reduction(
            name, dtype, src_dtype, reduction_type, self._simplify(index), value
        )

    def index_expr(self, index, dtype):
        return self._inner.index_expr(self._simplify(index), dtype)
