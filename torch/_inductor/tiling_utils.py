import dataclasses
import itertools
from collections import Counter, defaultdict
from typing import Callable, Literal, Optional, overload, TYPE_CHECKING, TypeVar, Union

import sympy

import torch
from torch._inductor import config
from torch._inductor.dependencies import index_vars_no_squeeze
from torch._inductor.utils import sympy_product, sympy_subs
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import Identity
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .virtualized import V


T = TypeVar("T")
U = TypeVar("U")


Split = tuple[sympy.Expr, ...]
VarsAndRanges = tuple[list[sympy.Symbol], list[sympy.Expr]]


loop_tiling_log = torch._logging.getArtifactLogger(__name__, "loop_tiling")
from torch.utils._sympy.functions import FloorDiv, ModularIndexing


if TYPE_CHECKING:
    from torch._inductor.scheduler import FusedSchedulerNode, SchedulerNode


def solve_for_zero(expr: sympy.Expr) -> Optional[sympy.Expr]:
    """
    Given an expr with a single free symbol, solve for a constant relation that would make
    this expression 0.
    """
    if expr.is_constant():
        return None
    elif isinstance(expr, FloorDiv):
        return None

    assert len(expr.free_symbols) == 1
    free_symbol = next(iter(expr.free_symbols))
    if isinstance(expr, ModularIndexing):
        out = try_solve(sympy.Eq(expr.args[0], expr.args[2]), free_symbol)
    else:
        out = try_solve(sympy.Eq(expr, 0), free_symbol)
    if not out or not out[1].is_constant():
        return None
    return out[1]


def solve_for_tiling(expr: sympy.Expr) -> Optional[sympy.Expr]:
    """
    Giving an expr with a single free symbol, try to find a tiling that would
    make the expression coalesced with respect to that symbol.

    Tiling an expression `x` by `y` means that the expression will now be indexed
    by both the original (x) and by (x * y). So we are looking for a
    multiplicative factor that will make ((x + 1) * y) - (x * y) == 1.

    To simplify things for sympy, we'll try just x * y == 1, check x(1) and x(0).
    """

    if len(expr.free_symbols) == 0:
        return None

    free_symbol = next(iter(expr.free_symbols))

    def _solve_simple_expr(expr: sympy.Expr) -> Optional[sympy.Expr]:
        assert not expr.has(ModularIndexing) and not expr.has(FloorDiv)
        if len(expr.free_symbols) != 1:
            return None

        out = try_solve(sympy.Eq(expr, 1), free_symbol)
        if not out or not out[1].is_constant():
            return None
        return out[1]

    # Sympy solving is very limited with ModularIndexing and FloorDiv,
    # but good otherwise.
    if not expr.has(ModularIndexing) and not expr.has(FloorDiv):
        return _solve_simple_expr(expr)

    required_values = []
    eq_1_expressions = []

    # very piecemeal solution if ModularIndexing or FloorDiv involved.
    # Look for terms we'll try to make 0, and then other terms we'll try to make 1.
    # Expand as needed.
    for arg in sympy.Add.make_args(expr):
        # Try to make mul terms 0
        if isinstance(arg, sympy.Mul):
            seen = False
            # TODO - only need one of these to be solvable to zero
            #
            for mul_arg in arg.args:
                out = solve_for_zero(mul_arg)
                if out is None:
                    continue

                assert out.is_constant()
                seen = True
                required_values.append(out)

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

    # For the purposes of tiling/coalesced access, approximate ModularIndexing and FloorDiv
    # then check later
    # pyrefly: ignore  # missing-attribute
    eq_1_expr_simplified = eq_1_expr.replace(ModularIndexing, indexing_div_rep).replace(
        FloorDiv, indexing_div_rep
    )

    out = _solve_simple_expr(eq_1_expr_simplified)
    # since we approximated FloorDiv/ModularIndexing, double check here
    if not out or sympy_subs(eq_1_expr, {free_symbol: out}) != 1:
        return None

    required_values.append(out)

    if len(OrderedSet(required_values)) == 1:
        return required_values[0]

    return None


def find_coalesced_var(
    index: sympy.Expr, var_ranges: dict[sympy.Expr, int]
) -> Optional[sympy.Expr]:
    """
    Try to find the symbol which coalesces this index
    """
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
            loop_tiling_log.info("zero division error %s %s", index, variables)
            continue
        if new_val - zero_index == 1:
            variables[v] = 2
            # in some more complex expressions, 0->1 will be coalesced,
            # but not 1->2
            if (sympy_subs(index, variables) - new_val) == 1:
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
    reads: dict[sympy.Expr, OrderedSet[str]]
    writes: dict[sympy.Expr, OrderedSet[str]]
    var_ranges: dict[sympy.Symbol, int]


@overload
def get_pw_red_splits(
    n: "SchedulerNode",
    pointwise_numel: sympy.Expr,
    red_numel: sympy.Expr,
    none_if_not_divisible: Literal[True],
) -> Optional[tuple[VarsAndRanges, VarsAndRanges]]: ...


@overload
def get_pw_red_splits(
    n: "SchedulerNode",
    pointwise_numel: sympy.Expr,
    red_numel: sympy.Expr,
    none_if_not_divisible: Literal[False] = False,
) -> tuple[VarsAndRanges, VarsAndRanges]: ...


def get_pw_red_splits(
    n: "SchedulerNode",
    pointwise_numel: sympy.Expr,
    red_numel: sympy.Expr,
    none_if_not_divisible: bool = False,
) -> Optional[tuple[VarsAndRanges, VarsAndRanges]]:
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
        i -= 1

    if i >= 0:
        pw_splits = n._body.sizes[0][0:i]
        iter_vars = n._body.iter_vars[0:i]

        red_splits = n._body.sizes[0][i:]
        red_vars = n._body.iter_vars[i:]
        return (iter_vars, pw_splits), (red_vars, red_splits)  # type: ignore[return-value]

    if none_if_not_divisible:
        return None
    else:
        return (
            (n._body.iter_vars, n._body.sizes[0]),
            (n._body.reduce_vars, n._body.sizes[1]),
        )  # type: ignore[return-value]


class NodeSplitGetter:
    """
    Finds a Pointwise, Reduction Split that compatible with all nodes in a SchedulerNode.
    """

    def __init__(
        self,
        node: Union["FusedSchedulerNode", "SchedulerNode"],
    ):
        self.node = node
        self.pointwise_numel: sympy.Expr = node.group[1][0]
        self.red_numel: sympy.Expr = node.group[1][1]

        self.pw_split_options: dict[int, OrderedSet[Split]] = defaultdict(OrderedSet)

        self.reduction_split: Split = ()
        self.all_node_sizes: OrderedSet[tuple[Split, Split]] = OrderedSet()

        fused_group = node.group[1]
        for n in reversed(node.get_nodes()):
            if not isinstance(n, torch._inductor.scheduler.SchedulerNode):
                continue

            # if we can't split the pw ranges into a (pw, red) split,
            # dont add as a split option, but do make sure we check that this size
            # is splittable
            maybe_splits = get_pw_red_splits(
                n, self.pointwise_numel, self.red_numel, none_if_not_divisible=True
            )
            if maybe_splits is None:
                self.all_node_sizes.add(n._body.sizes)
                continue

            (_, n_pw_splits), (_, n_red_splits) = maybe_splits

            # fill in reduction size
            n_pw_splits, n_red_splits = (
                torch._inductor.codegen.simd.SIMDKernel.prepare_split_iteration_lengths(
                    fused_group, (n_pw_splits, n_red_splits), self.red_numel
                )
            )

            self.pw_split_options[len(n_pw_splits)].add(tuple(n_pw_splits))

            # initially, we are just going to do a single reduction split since
            # reduction tiling is off by default. even if we miss a reduction split,
            # we can recover it in the split var analysis.
            # TODO: an earlier version for this code tried to iteratively try the maximum number
            # of split vars, by iterating over both pointwise and reduction. but not worth
            # the complexity yet.

            if n_red_splits != ():
                self.reduction_split = (sympy_product(n_red_splits),)

            n_size = (tuple(n_pw_splits), tuple(n_red_splits))
            self.all_node_sizes.add(n_size)

        self.seen_pw_splits: OrderedSet[Split] = OrderedSet()

    def get_node_splits(self) -> tuple[Split, Split]:
        """
        Get a compatible pointwise, reduction split of the node
        """

        if len(self.all_node_sizes) == 1:
            return next(iter(self.all_node_sizes))

        max_pw_split = max(self.pw_split_options.keys())
        for pw_split_len in range(max_pw_split, 0, -1):
            for pw_split in self.pw_split_options[pw_split_len]:
                if out := self.try_split(pw_split, self.reduction_split):
                    return out

            # combine dims for next round
            for pw_split in self.pw_split_options[pw_split_len]:
                for i in range(len(pw_split) - 1):
                    new_split = tuple(
                        pw_split[0:i]
                        + (sympy_product(pw_split[i : i + 2]),)
                        + pw_split[i + 2 :]
                    )
                    self.pw_split_options[len(new_split)].add(new_split)

        # if for whatever reason we couldn't split above, return default split
        return ((self.pointwise_numel,), (self.red_numel,))

    def try_split(self, pw: Split, red: Split) -> Optional[tuple[Split, Split]]:
        """
        See if this split is compatible, and potentially returning a longer split
        than the input.
        """

        from torch._inductor.codegen.simd import CantSplit, SIMDKernel

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
            pw_group_splits = splits[: len(pw)]
            # if we had to divide a variable into two to do this split,
            # then lets try the larger, induced split.
            # e.g. splitting (12, 2) into (2, 12) will split the first var into:
            # (2, 6) and produce an overall split of (2, 6, 2)
            flattened_pw_splits = tuple(itertools.chain.from_iterable(pw_group_splits))
            if flattened_pw_splits != pw:
                if out := self.try_split(flattened_pw_splits, red):
                    return out

        return pw, red


def apply_var_mapping(
    iter_vars: list[sympy.Symbol],
    red_vars: list[sympy.Symbol],
    norm_pw_vars: list[sympy.Symbol],
    norm_red_vars: list[sympy.Symbol],
    new_ranges: list[list[sympy.Expr]],
    return_getters_groups: list[list[Callable[[list[sympy.Expr]], sympy.Expr]]],
) -> dict[sympy.Symbol, sympy.Expr]:
    """Maps original variables to expressions using normalized variables."""

    # the output of split_iteration_range is a new_ranges, return_getters_groups
    # new_ranges is a flattened list of ranges corresponding to the new pw and red vars
    # for example, taking in pw vars of range (6, 6) to normalized range [36],
    # new_ranges would be [[6, 6]]
    # There is a return_getter callable for each input iter_var and red_vars.
    # if you flatten out all of the ranges, and create a variable for each index,
    # then applying the flattening vars to the callables in return_getters_groups
    # gives you the mapping from input vars -> flattened vars.
    # From there, we can compute the output, normalized variables.
    # For instance [6, 6] corresponding to flat vars v0, v1 will be
    # v0 + 6 * v1

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
    for i, (group, var_group) in enumerate(
        zip(apply_groups, (iter_vars, red_vars), strict=True)
    ):
        # if the node has sizes (p0, 1) and the fused node is (p0, r0)
        # the reduction var gets filled in for split_iteration_range
        if len(group) != len(var_group):
            assert i == 1
            assert len(var_group) == 0
            continue

        iter_vars_to_flat_vars.update({v: g for g, v in zip(group, var_group)})

    count = 0
    flat_vars_to_new_vars = {}
    for new_range, new_var in zip(
        new_ranges, norm_pw_vars + norm_red_vars, strict=True
    ):
        range_vars = []
        for i in range(len(new_range)):
            range_vars.append(flat_vars[count])
            count += 1

        prod = 1
        for i in range(len(new_range) - 1, -1, -1):
            flat_vars_to_new_vars[range_vars[i]] = new_var * prod
            prod = new_range[i] * prod

    return {
        k: sympy_subs(v, flat_vars_to_new_vars)
        for k, v in iter_vars_to_flat_vars.items()
    }


def extract_normalized_read_writes(
    node: Union["FusedSchedulerNode", "SchedulerNode"],
) -> Optional[FusedNormalizedReadsWrites]:
    """Extracts index variables, reduce variables, read/write expressions, and variable ranges from a fused node."""
    reads: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)
    writes: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)

    all_output_names = node.get_buffer_names()
    op_names = node.get_operation_names()
    outputs: OrderedSet[str] = OrderedSet()
    removed_buffers: OrderedSet[str] = OrderedSet()
    for buf_name in all_output_names:
        if V.graph.scheduler.can_buffer_be_removed_through_fusion(buf_name, op_names):
            removed_buffers.add(buf_name)
        else:
            outputs.add(buf_name)

    inputs = OrderedSet(
        dep.name for dep in node.read_writes.reads if dep.name not in removed_buffers
    )

    pointwise_numel: sympy.Expr = node.group[1][0]
    red_numel: sympy.Expr = node.group[1][1]

    # TODO - a few dynamic shapes issues to resolve
    if any(
        (isinstance(var, sympy.Expr) and not var.is_constant())
        for var in (pointwise_numel, red_numel)
    ):
        return None

    pw_splits, red_splits = NodeSplitGetter(node).get_node_splits()

    # lets use different prefix (`n`) to distinguish
    (norm_pw_vars, norm_red_vars), ranges = index_vars_no_squeeze(
        pw_splits, red_splits, prefix="n"
    )

    for n in list(node.get_nodes()):
        if not isinstance(n, torch._inductor.scheduler.SchedulerNode):
            continue

        body = n._body

        # TODO - not handled well. indirect loads will not be coalesced,
        # need to account for that in analysis.
        if body.indirect_vars:
            return None

        n_reads: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)
        n_writes: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)

        # TODO - will the names for all the inputs/outputs accurately
        # reflect mutation, or do I need to remap with mutation_real_name
        for inp in inputs:
            for expr in body.get_all_read_expr(inp):
                n_reads[expr].add(inp)

        for out in outputs:
            for expr in body.get_all_write_expr(out):
                n_writes[expr].add(out)

        if not n_reads and not n_writes:
            continue

        (iter_vars, n_pw_splits), (red_vars, n_red_splits) = get_pw_red_splits(
            n, pointwise_numel, red_numel
        )

        groups = pw_splits + red_splits
        lengths = (n_pw_splits, (n_red_splits))
        lengths = (
            torch._inductor.codegen.simd.SIMDKernel.prepare_split_iteration_lengths(
                groups, lengths, red_numel
            )
        )
        new_ranges, return_getters_groups = (
            torch._inductor.codegen.simd.SIMDKernel._split_iteration_ranges(
                groups, lengths
            )
        )
        var_map = apply_var_mapping(
            iter_vars,
            red_vars,
            norm_pw_vars,
            norm_red_vars,
            new_ranges,
            return_getters_groups,
        )

        # We create Identity sympy.Functions to prevent expansion to int64,
        # unwrap for tiling analysis.
        def remove_identity(expr: sympy.Expr) -> sympy.Expr:
            return expr.replace(Identity, lambda x: x)

        n_reads_new = {
            sympy_subs(remove_identity(read), var_map): v for read, v in n_reads.items()
        }
        n_writes_new = {
            sympy_subs(remove_identity(write), var_map): v
            for write, v in n_writes.items()
        }

        for expr, buf_names in n_reads_new.items():
            reads[expr] |= buf_names

        for expr, buf_names in n_writes_new.items():
            writes[expr] |= buf_names

    reads = {
        V.graph.sizevars.simplify_with_ranges(r, ranges): v for r, v in reads.items()
    }
    writes = {
        V.graph.sizevars.simplify_with_ranges(w, ranges): v for w, v in writes.items()
    }

    fused_out = FusedNormalizedReadsWrites(
        norm_pw_vars,  # type: ignore[arg-type]
        norm_red_vars,  # type: ignore[arg-type]
        reads,
        writes,
        ranges,
    )
    loop_tiling_log.info("Normalized Fused reads: %s", fused_out)
    return fused_out


def get_score(addr: sympy.Expr, var_ranges: dict[sympy.Symbol, int]) -> int:
    """
    Score addr according to its approximate size
    """

    # TODO - deduplicate with candidate_tilings
    var_sizes = []
    for v in addr.free_symbols:
        v_size = var_ranges.get(v)
        # TODO - reason about indirect vars
        if not symbol_is_type(v, SymT.INDIRECT) and v_size is not None:
            var_sizes.append(v_size)
    from .virtualized import V

    return V.graph.sizevars.atomically_apply_size_hint(
        sympy_product(var_sizes), fallback=config.unbacked_symint_fallback
    )


def get_hint(v: Union[sympy.Expr, int]) -> int:
    if isinstance(v, int):
        return v
    else:
        return V.graph.sizevars.size_hint(v, fallback=config.unbacked_symint_fallback)


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
    # Var -> Memory Score - not strictly the amount of memory
    # because we multiply writes x2
    # TODO: separate into dataclass that olds mem, dtype, is_write
    coalesced_by_var: dict[sympy.Expr, int]

    norm_read_writes: FusedNormalizedReadsWrites

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

    norm_read_writes = extract_normalized_read_writes(fused_node)

    if norm_read_writes is None:
        return None

    reads = norm_read_writes.reads
    writes = norm_read_writes.writes
    var_ranges = norm_read_writes.var_ranges

    coalesced_by_var: dict[sympy.Symbol, int] = Counter()
    uncoalesced_addrs: dict[sympy.Expr, int] = Counter()

    for is_read, (memory_expr, buf_names) in itertools.chain(
        ((True, item) for item in reads.items()),
        ((False, item) for item in writes.items()),
    ):
        # skip memory deps with indirect vars - todo: better handling
        indirect_expr = bool(
            memory_expr.free_symbols - norm_read_writes.var_ranges.keys()
        )

        if indirect_expr:
            continue

        size = get_score(memory_expr, var_ranges)
        if size == 0:
            continue

        maybe_coalesced_var = find_coalesced_var(memory_expr, var_ranges)

        byte_multipler = 0
        for buf_name in buf_names:
            if buf := V.graph.try_get_buffer(buf_name):
                byte_multipler += buf.dtype.itemsize

        # coalesced writes more important
        byte_multipler *= 1 if is_read else 2

        if maybe_coalesced_var:
            coalesced_by_var[maybe_coalesced_var] += size * byte_multipler
        else:
            uncoalesced_addrs[memory_expr] += size * byte_multipler

    if not uncoalesced_addrs:
        return CoalesceVarAnalysis(
            coalesced_by_var=coalesced_by_var, norm_read_writes=norm_read_writes
        )

    # map from var -> tiling -> total_score
    tiling_scores: dict[sympy.Expr, dict[int, int]] = defaultdict(Counter)

    for uncoalesced_expr, addr_score in uncoalesced_addrs.items():
        expr_subs = dict.fromkeys(uncoalesced_expr.free_symbols, 0)
        for v in uncoalesced_expr.free_symbols:
            # skip non iter/reduce var variables
            if v not in var_ranges:
                continue
            # skip small addrs
            if addr_score == 0:
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
            if not V.graph.sizevars.statically_known_lt(tiling_factor, var_ranges[v]):
                continue

            # TODO - if a var is in the middle, such as [n0, n1, n2]
            # n1 can can be split beyond range

            MIN_TILING_BLOCK = 8
            if not all(
                V.graph.sizevars.statically_known_lt(MIN_TILING_BLOCK, block)
                for block in (tiling_factor, var_ranges[v] // tiling_factor)
            ):
                continue

            tiling_scores[v][tiling_factor] += addr_score

    if len(tiling_scores) == 0:
        return CoalesceVarAnalysis(
            coalesced_by_var=coalesced_by_var, norm_read_writes=norm_read_writes
        )

    best_tiling: Optional[tuple[sympy.Expr, int]] = None
    best_tiling_score = 0

    for var, tiling_counter in tiling_scores.items():
        for tile, tile_score in tiling_counter.items():
            if tile_score > best_tiling_score:
                best_tiling = (var, tile)
                best_tiling_score = tile_score

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
