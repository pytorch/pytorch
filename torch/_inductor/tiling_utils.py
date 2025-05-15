import dataclasses
import itertools
from collections import Counter, defaultdict
from operator import is_
from re import A
from typing import Optional, TYPE_CHECKING, Union, Sequence

import sympy

import torch


from torch._inductor import config
from torch._inductor.dependencies import index_vars_no_squeeze
from torch._inductor.utils import sympy_product, sympy_subs
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .virtualized import V
Split = tuple[sympy.Expr]


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


def solve_for_tiling(expr: sympy.Expr) -> Optional[sympy.Expr]:
    """
    Giving an expr with a single free symbol, try to find a tiling that would
    make the expression coalesced with respect to that symbol.
    """
    if len(expr.free_symbols) == 0:
        return None

    assert len(expr.free_symbols) == 1
    free_symbol = next(iter(expr.free_symbols))

    # Sympy solving is very limited with ModularIndexing and FloorDiv,
    # but good otherwise.
    if not expr.has(ModularIndexing) and not expr.has(FloorDiv):
        expr_plus_one = sympy_subs(expr, {free_symbol: free_symbol + 1})

        diff = expr_plus_one - expr
        if diff.is_constant() and diff >= 0:
            # breakpoint()
            return diff

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

    def indexing_div_rep(
        x: sympy.Expr,
        y: sympy.Expr,
        z: Optional[sympy.Expr] = None,
    ) -> sympy.Expr:
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
    variables: dict[sympy.Symbol, int] = {}
    for v in index.free_symbols:
        if v in var_ranges:
            variables[v] = 0
        else:
            variables[v] = get_hint(v)

    zero_index = sympy_subs(index, variables)
    for v in var_ranges.keys():
        variables[v] = 1
        try:
            new_val = sympy_subs(index, variables)
        except ZeroDivisionError:
            continue
        if new_val - zero_index == 1:
            return v
        variables[v] = 0

    return None


@dataclasses.dataclass(frozen=True)
class FusedNormalizedReadsWrites:
    """
    Normalized reads and writes for nodes in the same FusedSchedulerNode.
    """

    index_vars: OrderedSet[sympy.Symbol]
    reduce_vars: OrderedSet[sympy.Symbol]
    reads: OrderedSet[sympy.Expr]
    writes: OrderedSet[sympy.Expr]
    var_ranges: dict[sympy.Symbol, int]



def get_pw_red_splits(
    n: "SchedulerNode", pointwise_numel: sympy.Expr, red_numel: sympy.Expr
) -> tuple[
    tuple[list[sympy.Symbol], list[int]], tuple[list[sympy.Symbol], list[int]]
]:
    if n.is_reduction() or sympy_product(n._body.sizes[0]) == pointwise_numel:
        return (
            (n._body.iter_vars, n._body.sizes[0]),
            (n._body.reduce_vars, n._body.sizes[1]),
        )  # type: ignore[return-value]

    assert sympy_product(n._body.sizes[0]) == pointwise_numel * red_numel  # type: ignore[operator]
    i = len(n._body.sizes[0]) - 1
    prod = 1
    while i >= 0:
        prod *= n._body.sizes[0][i]
        if prod == red_numel:
            break

    if i >= 0:
        pw_splits = n._body.sizes[0][0:i]
        iter_vars = n._body.iter_vars[0:i]

        red_splits = n._body.sizes[0][i:]
        red_vars = n._body.iter_vars[i:]
        return (iter_vars, pw_splits), (red_vars, red_splits)  # type: ignore[return-value]

    # TODO - handle
    raise RuntimeError(
        f"Unhandled node: size: {n._body.sizes}, pw: {pointwise_numel}, red: {red_numel}"
    )


class NodeSplitGetter():

    def __init__(self, node: Union["FusedSchedulerNode", "SchedulerNode"],):
        from torch._inductor.codegen.simd import SIMDKernel, CantSplit
        self.node = node
        self.pointwise_numel: sympy.Expr = node.group[1][0]
        self.red_numel: sympy.Expr = node.group[1][1]

        self.pw_split_options: dict[int, OrderedSet[Split]] = defaultdict(OrderedSet)

        self.reduction_split: Split = ()

        if self.pointwise_numel == 1:
            self.pw_split_options[0].add(())

        self.all_node_sizes: OrderedSet[tuple[Split, Split]] = OrderedSet()
        
        fused_group = node.group[1]
        for n in reversed(node.get_nodes()):
            if not isinstance(n, torch._inductor.scheduler.SchedulerNode):
                continue

            (_, n_pw_splits), (_, n_red_splits) = get_pw_red_splits(n, self.pointwise_numel, self.red_numel)
                        
            # fill in reduction size
            n_pw_splits, n_red_splits = torch._inductor.codegen.simd.SIMDKernel.prepare_split_iteration_lengths(fused_group, (n_pw_splits, n_red_splits), self.red_numel)
            
            self.pw_split_options[len(n_pw_splits)].add(tuple(n_pw_splits))

            # initially, we are just going to do a single reduction split since
            # reduction tiling is off by default. even if we miss a reduction split, 
            # we can recover it in the split var analysis.
            # TODO: an earlier version fo this code tried to iteratively try the maximum number 
            # of split vars, by iterating over both pointwise and reduction. but not worth
            # the complexity yet.  

            if n_red_splits != ():
                self.reduction_split = (sympy_product(n_red_splits),)

            n_size = (tuple(n_pw_splits), tuple(n_red_splits))
            self.all_node_sizes.add(n_size)

        self.seen_pw_splits: OrderedSet[Split] = OrderedSet()
        
    def get_node_splits(self):
        max_pw_split = max(self.pw_split_options.keys())
        for pw_split_len in range(max_pw_split, 0, -1):
            for pw_split in self.pw_split_options[pw_split_len]:
                if out := self.try_split(pw_split, self.reduction_split):
                    return out

            # combine dims for next round
            for pw_split in self.pw_split_options[pw_split_len]:
                for i in range(len(pw_split) - 1):
                    new_split = tuple(
                        pw_split[0:i] + tuple([sympy_product(pw_split[i:i+2])]) + pw_split[i+2:]
                    )
                    self.pw_split_options[len(new_split)].add(new_split)

        # if for whatever reason we couldnt split above, return default split
        return ((self.pointwise_numel,), (self.red_numel,))

    def try_split(self, pw: Split, red: Split) -> Optional[tuple[Split, Split]]:
        from torch._inductor.codegen.simd import SIMDKernel, CantSplit

        if pw in self.seen_pw_splits:
            return None
        self.seen_pw_splits.add(pw)

        for n_pw, n_red in self.all_node_sizes:
            try:
                groups = pw + red
                lengths = (n_pw, n_red)
                splits, getters = SIMDKernel._split_iteration_ranges(groups, lengths)
            except CantSplit:
                return None

            assert len(getters) == 2
            pw_group_splits = splits[:len(pw)]
            # if we had to divide a variable into two to do this split, 
            # then lets try the larger, induced split.
            # e.g. splitting (12, 2) into (2, 12) will split the first var into:
            # (2, 6) and produce an overall split of (2, 6, 2)
            flattened_pw_splits = tuple(itertools.chain.from_iterable(pw_group_splits))
            if flattened_pw_splits != pw:
                if out := self.try_split(flattened_pw_splits, red):
                    return out
        
        return pw, red

def zip_equal(it1, it2):
    if len(it1) != len(it2):
        breakpoint()
        raise ValueError("Lengths of iterables are different")
    return zip(it1, it2)

def apply_var_mapping(iter_vars, red_vars, norm_pw_vars, norm_red_vars, new_ranges, return_getters_groups):
    """Maps original variables to expressions using normalized variables."""
    # Create flattened iteration variables
    num_vars = sum(len(s) for s in new_ranges)
    flat_vars = sympy.symbols(f"v_0:{num_vars}")
    count = 0

    if len(iter_vars) == 0 and len(red_vars) == 0:
        return {}

    assert len(new_ranges) == len(norm_pw_vars + norm_red_vars)
    apply_groups = []
    for group in return_getters_groups:
        apply_groups.append([g(flat_vars) for g in group])

    iter_vars_to_flat_vars = {}
    for i, (group, var_group) in enumerate(zip_equal(apply_groups, ((iter_vars, red_vars)))):

        # if the node has sizes (p0, 1) and the fused node is (p0, r0)
        # the reduction var gets filled in for split_iteration_range
        if len(group) != len(var_group):
            assert i == 1
            assert len(var_group) == 0
            continue

        for g, v in zip(group, var_group):
            iter_vars_to_flat_vars[v] = g

    count = 0
    flat_vars_to_new_vars = {}
    for new_range, new_var in zip_equal(new_ranges, norm_pw_vars + norm_red_vars):
        range_vars = []
        for i in range(len(new_range)):
            range_vars.append(flat_vars[count])
            count += 1

        prod = 1
        for i in range(len(new_range) -1, -1, -1):
            flat_vars_to_new_vars[range_vars[i]] = new_var * prod
            prod = new_range[i] * prod

    final_dict = {k: sympy_subs(v, flat_vars_to_new_vars) for k, v in iter_vars_to_flat_vars.items()}
    return final_dict


def _extract_fused_node_meta(
    node: Union["FusedSchedulerNode", "SchedulerNode"],
) -> FusedNormalizedReadsWrites:

    """Extracts index variables, reduce variables, read/write expressions, and variable ranges from a fused node."""
    reads: OrderedSet[sympy.Expr] = OrderedSet()
    writes: OrderedSet[sympy.Expr] = OrderedSet()

    outputs = node.get_buffer_names()
    inputs = OrderedSet(dep.name for dep in node.read_writes.reads)

    pw_splits, red_splits = NodeSplitGetter(node).get_node_splits()

    # lets use different prefix (`n`) to distinguish
    (norm_pw_vars, norm_red_vars), ranges = index_vars_no_squeeze(
        pw_splits, red_splits, prefix="n"
    )
    node = node
    pointwise_numel: sympy.Expr = node.group[1][0]
    red_numel: sympy.Expr = node.group[1][1]
    
    for n in (list(node.get_nodes())):
        if not isinstance(n, torch._inductor.scheduler.SchedulerNode):
            continue

        body = n._body
        n_reads: OrderedSet[sympy.Expr] = OrderedSet()
        n_writes: OrderedSet[sympy.Expr] = OrderedSet()
        for inp in inputs:
            n_reads |= body.get_all_read_expr(inp)
        for out in outputs:
            n_writes |= body.get_all_write_expr(out)

        (iter_vars, n_pw_splits), (red_vars, n_red_splits) = get_pw_red_splits(n, pointwise_numel, red_numel)
       
        groups = pw_splits + red_splits
        lengths = (n_pw_splits, (n_red_splits))
        lengths = torch._inductor.codegen.simd.SIMDKernel.prepare_split_iteration_lengths(groups, lengths, red_numel)
        new_ranges, return_getters_groups = torch._inductor.codegen.simd.SIMDKernel._split_iteration_ranges(groups, lengths)
        var_map = apply_var_mapping(
            iter_vars, red_vars, 
            norm_pw_vars, norm_red_vars, 
            new_ranges, return_getters_groups
        )

        n_reads_new = [sympy_subs(read, var_map) for read in n_reads]
        n_writes_new = [sympy_subs(read, var_map) for read in n_writes]

        reads |= n_reads_new
        writes |= n_writes_new

    return FusedNormalizedReadsWrites(
        norm_pw_vars,  # type: ignore[arg-type]
        norm_red_vars,  # type: ignore[arg-type]
        reads,
        writes,
        ranges,
    )


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

    return V.graph.sizevars.atomically_apply_size_hint(
        sympy_product(var_sizes), fallback=config.unbacked_symint_fallback
    )


@dataclasses.dataclass(frozen=True)
class VarTiling:
    """
    Tiling of a var by `tiling_factor` that yields additional coalesced mem accesses by `benefit_score`
    """

    var: sympy.Symbol
    tiling_factor: int
    score: int


def get_hint(v: Union[sympy.Expr, int]) -> int:
    if isinstance(v, int):
        return v
    else:
        return V.graph.sizevars.size_hint(v, fallback=config.unbacked_symint_fallback)


@dataclasses.dataclass(frozen=True)
class CoalesceVarAnalysis:
    coalesced_by_var: dict[sympy.Expr, int]

    norm_read_writes: FusedNormalizedReadsWrites

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

    norm_read_writes = _extract_fused_node_meta(fused_node)

    reads = norm_read_writes.reads
    writes = norm_read_writes.writes
    var_ranges = norm_read_writes.var_ranges

    coalesced_by_var: dict[sympy.Symbol, int] = Counter()
    uncoalesced_addrs: dict[sympy.Expr, int] = {}

    for memory_expr in itertools.chain(reads, writes):
        size = get_score(memory_expr, var_ranges)
        maybe_coalesced_var = find_coalesced_var(memory_expr, var_ranges)
        if maybe_coalesced_var:
            coalesced_by_var[maybe_coalesced_var] += size
        else:
            uncoalesced_addrs[memory_expr] = size

    if not uncoalesced_addrs:
        return CoalesceVarAnalysis(
            coalesced_by_var=coalesced_by_var, norm_read_writes=norm_read_writes
        )

    # map from var -> tiling -> total_score
    potential_tiling_scores: dict[sympy.Expr, dict[int, int]] = defaultdict(Counter)

    for uncoalesced_expr, addr_score in uncoalesced_addrs.items():
        expr_subs = dict.fromkeys(uncoalesced_expr.free_symbols, 0)
        for v in uncoalesced_expr.free_symbols:
            # skip non iter/reduce var variables
            if v not in var_ranges:
                continue
            del expr_subs[v]
            single_var_expr = sympy_subs(uncoalesced_expr, expr_subs)
            expr_subs[v] = 0
            tiling_factor = solve_for_tiling(single_var_expr)
            if (
                tiling_factor is None
                or not tiling_factor.is_constant()
                or not tiling_factor.is_integer
            ):
                continue

            tiling_factor = int(tiling_factor)
            MIN_TILING_BLOCK = 4
            if any(
                (get_hint(b) < MIN_TILING_BLOCK)
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
        return CoalesceVarAnalysis(
            coalesced_by_var=coalesced_by_var, norm_read_writes=norm_read_writes
        )

    # TODO - for strictly pointwise fusions,
    # we can consider just swizzling the var if the var we are going to tile
    # does not coalesce a significant portion of global reads
    # TODO - could also prefer index var splits to reduction, better tested
    return CoalesceVarAnalysis(
        coalesced_by_var=coalesced_by_var,
        norm_read_writes=norm_read_writes,
        suggested_split=VarTiling(best_tiling[0], best_tiling[1], best_tiling_score),
    )
