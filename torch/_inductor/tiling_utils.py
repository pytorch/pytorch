import dataclasses
import functools
import itertools
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Callable, Optional, TYPE_CHECKING, TypeVar, Union

import sympy

import torch
from torch._inductor.dependencies import index_vars_no_squeeze
from torch._inductor.utils import sympy_product, sympy_subs
from torch.utils._ordered_set import OrderedSet

from .virtualized import V


T = TypeVar("T")
U = TypeVar("U")


Split = tuple[sympy.Expr, ...]

loop_tiling_log = torch._logging.getArtifactLogger(__name__, "loop_tiling")


if TYPE_CHECKING:
    from torch._inductor.scheduler import FusedSchedulerNode, SchedulerNode


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


def get_pw_red_splits(
    n: "SchedulerNode", pointwise_numel: sympy.Expr, red_numel: sympy.Expr
) -> tuple[tuple[list[sympy.Symbol], list[int]], tuple[list[sympy.Symbol], list[int]]]:
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

    # TODO - handle, not sure if possible
    raise RuntimeError(
        f"Unhandled node: size: {n._body.sizes}, pw: {pointwise_numel}, red: {red_numel}"
    )


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

            (_, n_pw_splits), (_, n_red_splits) = get_pw_red_splits(
                n, self.pointwise_numel, self.red_numel
            )

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
            # TODO: an earlier version fo this code tried to iteratively try the maximum number
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

        # if for whatever reason we couldnt split above, return default split
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


if sys.version_info >= (3, 10):
    # On Python 3.10+ we can use zip(strict=True)
    zip_equal = functools.partial(zip, strict=True)
else:
    # Fallback for older versions
    def zip_equal(it1: Iterable[T], it2: Iterable[U]) -> Iterator[tuple[T, U]]:
        """
        Zip two iterables, raising ValueError if their lengths differ.
        """
        if len(it1) != len(it2):
            raise ValueError(f"Lengths differ: {len(it1)} != {len(it2)}")
        return zip(it1, it2)


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
        zip_equal(apply_groups, ((iter_vars, red_vars)))
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
    for new_range, new_var in zip_equal(new_ranges, norm_pw_vars + norm_red_vars):
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
) -> FusedNormalizedReadsWrites:
    """Extracts index variables, reduce variables, read/write expressions, and variable ranges from a fused node."""
    reads: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)
    writes: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)

    all_output_names = node.get_buffer_names()
    op_names = node.get_operation_names()
    outputs = OrderedSet(
        buf
        for buf in all_output_names
        if not V.graph.scheduler.can_buffer_be_removed_through_fusion(buf, op_names)
    )
    inputs = OrderedSet(dep.name for dep in node.read_writes.reads)

    pw_splits, red_splits = NodeSplitGetter(node).get_node_splits()

    # lets use different prefix (`n`) to distinguish
    (norm_pw_vars, norm_red_vars), ranges = index_vars_no_squeeze(
        pw_splits, red_splits, prefix="n"
    )
    node = node
    pointwise_numel: sympy.Expr = node.group[1][0]
    red_numel: sympy.Expr = node.group[1][1]

    for n in list(node.get_nodes()):
        if not isinstance(n, torch._inductor.scheduler.SchedulerNode):
            continue

        body = n._body
        n_reads: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)
        n_writes: dict[sympy.Expr, OrderedSet[str]] = defaultdict(OrderedSet)

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

        n_reads_new = {sympy_subs(read, var_map): v for read, v in n_reads.items()}
        n_writes_new = {sympy_subs(write, var_map): v for write, v in n_writes.items()}

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
