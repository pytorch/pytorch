import dataclasses
import itertools
from collections import Counter, defaultdict
from typing import Optional, TYPE_CHECKING, Union

import sympy

from torch._inductor.utils import sympy_product, sympy_subs
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.symbol import symbol_is_type, SymT


if TYPE_CHECKING:
    from torch._inductor.scheduler import FusedSchedulerNode, SchedulerNode


def solve_for_zero(expr: sympy.Expr) -> Optional[tuple[sympy.Rel, sympy.Expr]]:
    """
    Given an expr with a single free symbol, solve for a constant relation that would make
    this expression 0.
    """
    if expr.is_constant() and not expr == 0:
        return None
    elif isinstance(expr, FloorDiv):
        return None

    assert len(expr.free_symbols) <= 1
    free_symbol = next(iter(expr.free_symbols))
    if isinstance(expr, ModularIndexing):
        out = try_solve(sympy.Eq(expr.args[0], expr.args[2]), free_symbol)
    else:
        out = try_solve(sympy.Eq(expr, 0), free_symbol)
    if not out or not out[1].is_constant():
        return None
    return out


def solve_for_tiling(expr: sympy.Expr) -> Optional[int]:
    """
    Giving an expr with a single free symbol, try to find a tiling that would
    make the expression coalesced with respect to that symbol.
    """
    assert len(expr.free_symbols) == 1
    free_symbol = next(iter(expr.free_symbols))

    # Sympy solving is very limited with ModularIndexing and FloorDiv,
    # but good otherwise.
    if not expr.has(ModularIndexing) and not expr.has(FloorDiv):
        expr_plus_one = sympy_subs(expr, {free_symbol: free_symbol + 1})
        out = try_solve(sympy.Eq(expr_plus_one - expr, 1), free_symbol)
        if not out or not out[1].is_constant():
            return None
        return out[1]

    required_values = []
    eq_1_expressions = []

    # very piecemeal solution if ModularIndexing or FloorDiv involved.
    # Expand as needed.
    for arg in sympy.Add.make_args(expr):
        # Try to make mul terms 0
        if isinstance(arg, sympy.Mul):
            seen = False
            # TODO - only need one of these to be solvable to zero
            for mul_arg in arg.args:
                out = solve_for_zero(mul_arg)
                if out is None or not out[1].is_constant():
                    continue

                seen = True
                required_values.append(out[1])

            if not seen:
                return None
        else:
            eq_1_expressions.append(arg)

    if not eq_1_expressions:
        return None

    eq_1_expr = sum(eq_1_expressions)

    def indexing_div_rep(x, y, z=None):
        return x / y

    is_non_differentiable = eq_1_expr.has(ModularIndexing) or eq_1_expr.has(
        ModularIndexing
    )
    # For the purposes of tiling/coalesced access, we can treat ModularIndexing and FloorDiv equivalently
    eq_1_expr = eq_1_expr.replace(ModularIndexing, indexing_div_rep).replace(
        FloorDiv, indexing_div_rep
    )
    out = try_solve(sympy.Eq(eq_1_expr, 1), free_symbol)
    if out is None or not out[1].is_constant():
        return None

    # since we approximated FloorDiv/ModularIndexing, double check here
    if (
        is_non_differentiable
        and not (sympy_subs(eq_1_expr, {free_symbol: out[1]})) == 1
    ):
        return None

    required_values.append(out[1])

    if len(OrderedSet(required_values)) == 1:
        return required_values[0]

    return None


def find_coalesced_var(
    index: sympy.Expr, var_ranges: dict[sympy.Expr, int]
) -> Optional[sympy.Expr]:
    """
    Try to find the symbol which coalesces this index
    """
    # TODO - not sure what to do with indirect variable
    top_level_terms = sympy.Add.make_args(index)
    for v in var_ranges:
        if v in top_level_terms:
            return v

    # Approximate analysis by evaluating at 1 and 0
    variables = dict.fromkeys(index.free_symbols, 0)
    zero_index = sympy_subs(index, variables)
    for v in var_ranges.keys():
        variables[v] = 1
        new_val = sympy_subs(index, variables)
        if new_val - zero_index == 1:
            return v
        variables[v] = 0

    return None


def _extract_fused_node_meta(
    node: Union["FusedSchedulerNode", "SchedulerNode"],
) -> tuple[
    OrderedSet[sympy.Symbol],
    OrderedSet[sympy.Symbol],
    OrderedSet[sympy.Expr],
    OrderedSet[sympy.Expr],
    dict[sympy.Symbol, int],
]:
    """Extracts index variables, reduce variables, read/write expressions, and variable ranges from a fused node."""
    reads: OrderedSet[sympy.Expr] = OrderedSet()
    writes: OrderedSet[sympy.Expr] = OrderedSet()
    all_index_vars: OrderedSet[sympy.Symbol] = OrderedSet()
    all_reduce_vars: OrderedSet[sympy.Symbol] = OrderedSet()
    var_ranges: dict[sympy.Symbol, int] = {}

    outputs = node.get_buffer_names()
    inputs = OrderedSet(dep.name for dep in node.read_writes.reads)

    for n in node.get_nodes():
        body = n._body
        all_index_vars |= body.iter_vars
        all_reduce_vars |= body.reduce_vars
        var_ranges.update(body.var_ranges)

        for inp in inputs:
            reads |= body.get_all_read_expr(inp)
        for out in outputs:
            writes |= body.get_all_write_expr(out)

    return all_index_vars, all_reduce_vars, reads, writes, var_ranges


def get_score(addr: sympy.Expr, var_ranges: dict[sympy.Symbol, int]) -> int:
    """
    Score addr according to its approximate size
    """

    # TODO - deduplicate with candidate_tilings
    var_sizes = []
    for v in addr.free_symbols:
        v_size = var_ranges.get(v, None)
        if not symbol_is_type(v, SymT.INDIRECT) and v_size is not None:
            var_sizes.append(v_size)
    from .virtualized import V

    return V.graph.sizevars.size_hint(sympy_product(var_sizes))


@dataclasses.dataclass(frozen=True)
class VarTiling:
    """
    Tiling of a var by `tiling_factor` that yields additional coalesced mem accesses by `benefit_score`
    """

    var: sympy.Symbol
    tiling_factor: int
    score: int


@dataclasses.dataclass(frozen=True)
class CoalesceVarAnalysis:
    coalesced_by_var: dict[sympy.Expr, int]

    # Expression, split, score
    suggested_split: Optional[VarTiling] = None


def analyze_memory_coalescing(
    fused_node: Union["FusedSchedulerNode", "SchedulerNode"],
) -> Optional[CoalesceVarAnalysis]:
    """
    Find variables that coalesce the reads and writes and score the total size.

    If uncoalesced memory expressions are found, look for additionally tiling of variables
    which will coalesce memory accesses.

    For instance - for the following expression:

    (32*p0) // 2048

    Tiling p0 by 64 will make this expression coalesced.
    """

    _, _, reads, writes, var_ranges = _extract_fused_node_meta(fused_node)

    coalesced_by_var = Counter()
    uncoalesced_addrs: dict[sympy.Expr, int] = {}

    for memory_expr in itertools.chain(reads, writes):
        size = get_score(memory_expr, var_ranges)
        maybe_coalesced_var = find_coalesced_var(memory_expr, var_ranges)
        if maybe_coalesced_var:
            coalesced_by_var[maybe_coalesced_var] += size
        else:
            uncoalesced_addrs[memory_expr] = size

    if not uncoalesced_addrs:
        return CoalesceVarAnalysis(coalesced_by_var=coalesced_by_var)

    # map from var -> tiling -> total_score
    potential_tiling_scores: dict[sympy.Expr, dict[int, int]] = defaultdict(Counter)

    for uncoalesced_expr, addr_score in uncoalesced_addrs.items():
        expr_subs = {v: 0 for v in uncoalesced_expr.free_symbols}
        for v in uncoalesced_expr.free_symbols:
            del expr_subs[v]
            single_var_expr = sympy_subs(uncoalesced_expr, expr_subs)
            expr_subs[v] = 0
            tiling_factor = solve_for_tiling(single_var_expr)
            if tiling_factor is None or not tiling_factor.is_constant():
                continue

            MIN_TILING_BLOCK = 4
            if any(
                (b < MIN_TILING_BLOCK)
                for b in (tiling_factor, var_ranges[v] // tiling_factor)
            ):
                continue

            potential_tiling_scores[v][tiling_factor] += addr_score

    best_tiling: Optional[tuple[sympy.Expr, int]] = None
    best_tiling_score = 0

    for var, tiling_counter in potential_tiling_scores.items():
        for tile, tile_score in tiling_counter.items():
            score = tile_score - coalesced_by_var[var]
            if score > best_tiling_score:
                best_tiling = (var, tile)
                best_tiling_score = score

    if best_tiling is None:
        return CoalesceVarAnalysis(coalesced_by_var=coalesced_by_var)

    # TODO - for strictly pointwise fusions,
    # we can consider just swizzling the var if the var we are going to tile
    # does not coalesce a significant portion of global reads
    # TODO - could also prefer index var splits to reduction, better tested
    return CoalesceVarAnalysis(
        coalesced_by_var=coalesced_by_var,
        suggested_split=VarTiling(best_tiling[0], best_tiling[1], score),
    )
