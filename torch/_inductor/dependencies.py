import abc
import dataclasses
import itertools
import logging
import re
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Optional, TypeVar, Union
from typing_extensions import Self
from unittest.mock import patch

import sympy

import torch
from torch._inductor.utils import get_free_symbols
from torch.fx.experimental.symbolic_shapes import free_symbols, free_unbacked_symbols
from torch.utils._ordered_set import OrderedSet

from ..utils._sympy.symbol import make_symbol, SymT
from .codegen.common import index_prevent_reordering
from .ops_handler import DefaultHandler
from .utils import (
    get_dtype_size,
    reduction_num_outputs,
    sympy_index_symbol,
    sympy_str,
    sympy_subs,
    VarRanges,
)
from .virtualized import ReductionType, V


T = TypeVar("T")

log = logging.getLogger(__name__)
is_indirect = re.compile(r"indirect|tmp").search


class Dep(abc.ABC):
    name: str
    index: sympy.Expr

    @abc.abstractmethod
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        pass

    @abc.abstractmethod
    def rename(self, renames: dict[str, str]) -> Self:
        pass

    @abc.abstractmethod
    def get_numel(self) -> sympy.Expr:
        pass

    @abc.abstractmethod
    def numbytes_hint(self) -> int:
        pass

    @abc.abstractmethod
    def has_unbacked_symbols(self) -> bool:
        pass

    @abc.abstractmethod
    def is_contiguous(self) -> bool:
        pass

    def normalize_with_stride_order(self, prefix: str = "t") -> Self:
        return self


@dataclasses.dataclass(frozen=True)
class MemoryDep(Dep):
    name: str
    index: sympy.Expr
    var_names: tuple[sympy.Symbol, ...]
    size: tuple[sympy.Expr, ...]
    mode: Optional[str] = None

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return (
            get_free_symbols(self.index, unbacked_only)
            | get_free_symbols(self.size, unbacked_only)
            | get_free_symbols(self.var_names, unbacked_only)
        )

    def __repr__(self) -> str:
        maybe_mode = ""
        if self.mode is not None:
            maybe_mode = f", {self.mode}"
        return f"MemoryDep({self.name!r}, {self.index}, {self.ranges}{maybe_mode})"

    @property
    def num_vars(self) -> int:
        return len(self.var_names)

    def decide_loop_order_to_match(self, other: "MemoryDep") -> Optional[list[int]]:
        """
        Can return None if not able to decide loop orders.
        """
        assert self.num_vars == other.num_vars

        # ignore broadcast for now since broadcast causes extra 0 strides
        # which makes it hard to decide the correct loop orders.
        if self.num_vars != len(self.index.free_symbols):
            return None
        if other.num_vars != len(other.index.free_symbols):
            return None

        # bail out if any size is 0 or 1
        # For size == 0, it's an empty tensor, any strides for that dimension
        # are equivalent. Skip for simplicity and it may not matter that much.
        #
        # For size == 1, it cause cause tie for strides of different dimensions.
        # Also when we first time create LoopBody in ComputedBuffer.simplify_and_reorder
        # we can dependencies.index_vars_squeeze which should already sqeeuze
        # the size == 1 dimensions.
        if any(s == 0 or s == 1 for s in itertools.chain(self.size, other.size)):
            return None

        # Extract strides for both expression
        self_strides = V.graph.sizevars.stride_hints(self.index, self.var_names)
        other_strides = V.graph.sizevars.stride_hints(other.index, other.var_names)

        # Even if the shape contains no 0/1, some complex index expression may
        # still have duplicate stride values. Here is an example:
        # https://gist.github.com/shunting314/511a7e1ec88aa2e1a8ec85d8445ab129
        # We don't reorder the loop for these cases for now, but in theory
        # we could improve the algorithm to detect the correct loop orders.
        if len(OrderedSet(self_strides)) != len(self_strides) or len(
            OrderedSet(other_strides)
        ) != len(other_strides):
            log.debug(
                "unable to decide loop order. self_dep=%s v.s. other_dep=%s, self_strides=%s v.s. other_strides=%s",
                self,
                other,
                self_strides,
                other_strides,
            )
            return None

        # May happen if self and other are as follows
        # MemoryDep('addmm_6', 393216*d0 + 768*d1 + d2, {d0: 16, d1: 512, d2: 768}, None)
        # MemoryDep('addmm_6', 98304*d0 + d1 + 768*d2, {d0: 64, d1: 768, d2: 128}, None)
        if OrderedSet(self_strides) != OrderedSet(other_strides):
            return None

        stride_to_index = {s: i for i, s in enumerate(self_strides)}
        order = [stride_to_index[s] for s in other_strides]

        assert OrderedSet(order) == OrderedSet(range(0, self.num_vars))
        return order

    def get_offset(self) -> sympy.Expr:
        """
        Return the offset by setting every variable to be 0.
        """
        return sympy_subs(self.index, dict.fromkeys(self.var_names, 0))

    def normalize(self) -> "MemoryDep":
        """
        Normalize by merging loops. The different to normalize_with_stride_order is,
        this method does not reorder loops while normalize_with_stride_order reorder
        loops based on stride order.
        """
        return MemoryDep(
            self.name,
            *_RecordLoadStoreInner._normalize(self.index, self.ranges),  # type: ignore[arg-type]
            self.mode,
        )

    def normalize_with_stride_order(self, prefix: str = "t") -> "MemoryDep":
        r"""
        Used to decide if two MemoryDep does not equal due to different loop orders.
        More specifically, when dep1 and dep2 are not equal, we can normalize
        both and check if they are equal after that. If yes, then the mismatch is
        caused by different loop orders.
        """
        # import here to avoid circular import
        from torch._inductor import ir

        strides = V.graph.sizevars.stride_hints(self.index, self.var_names)

        # pick a loop order with stride ordered decreasingly
        order = sorted(range(len(strides)), key=strides.__getitem__, reverse=True)
        stride_reorder = ir.same_reorder(order)
        sizes = self.size
        var_names = self.var_names

        new_reordered_sizes = stride_reorder(sizes)
        new_reordered_var_names = stride_reorder(var_names)

        new_simplified_sizes, reindex, _prune = V.graph.sizevars._simplify_loops(
            new_reordered_var_names,
            new_reordered_sizes,
            index_prevent_reordering(
                [self.index], new_reordered_var_names, new_reordered_sizes
            ),
        )

        # now let's create new symbols with the passed in prefix
        var_ranges, add_var = var_builder(prefix)
        replacement = dict(
            zip(
                new_reordered_var_names,
                reindex([add_var(x) for x in new_simplified_sizes]),
            )
        )
        new_index = sympy_subs(sympy.expand(self.index), replacement)  # type: ignore[arg-type] # next PR

        out = MemoryDep(
            self.name, new_index, tuple(var_ranges.keys()), tuple(var_ranges.values())
        )  # type: ignore[arg-type]
        return out

    @property
    def ranges(self) -> dict[sympy.Symbol, sympy.Expr]:
        """{c0: 128, c1: 512, ...}"""
        return dict(zip(self.var_names, self.size))

    def simplify_with_ranges(self) -> "MemoryDep":
        return MemoryDep(
            name=self.name,
            index=V.graph.sizevars.simplify_with_ranges(self.index, self.ranges),
            var_names=self.var_names,
            size=self.size,
            mode=self.mode,
        )

    def get_numel(self) -> sympy.Expr:
        if self.is_indirect():
            numel = V.graph.get_numel(self.name)
        else:
            vars: OrderedSet[sympy.Basic] = OrderedSet(self.index.free_symbols)
            numel = sympy.S.One
            for var, size in zip(self.var_names, self.size):
                if var in vars:
                    numel = numel * size
        return numel  # type: ignore[return-value]

    def rename(self, renames: dict[str, str]) -> "MemoryDep":
        if self.name in renames:
            return MemoryDep(
                renames[self.name],
                self.index,
                var_names=self.var_names,
                size=self.size,
                mode=self.mode,
            )
        return self

    def numbytes_hint(self) -> int:
        try:
            return V.graph.sizevars.size_hint(self.get_numel()) * get_dtype_size(
                V.graph.get_dtype(self.name)
            )
        except NotImplementedError:  # NoneLayout
            return 0

    def has_unbacked_symbols(self) -> bool:
        return len(free_unbacked_symbols(self.get_numel())) > 0

    def is_contiguous(self) -> bool:
        if isinstance(self.index, sympy.Integer):
            return True
        return isinstance(self.index, sympy.Symbol) and self.index in self.var_names

    def stride1_for_last_dim(self, result_for_complex_expression: bool = True) -> bool:
        """
        Whether the stride for the last dimension is 1.
        """
        # python test/inductor/test_torchinductor_opinfo.py -k test_comprehensive_masked_scatter_cuda_float16
        # will exercise thru this corner case.
        if len(self.var_names) == 0:
            return True

        terms = self.index.args if isinstance(self.index, sympy.Add) else [self.index]

        last_sym = self.var_names[-1]
        for term in terms:
            if term == last_sym:
                return True

            # Having a >1 stride for the last dimension is bad for perf
            # return False.
            if (
                isinstance(term, sympy.Mul)
                and len(term.args) == 2
                and term.args[1] == last_sym
                and isinstance(term.args[0], (int, sympy.Integer))
                and term.args[0] > 1
            ):
                return False

        return result_for_complex_expression

    def is_scalar(self) -> bool:
        if isinstance(self.index, sympy.Symbol):
            return self.index not in self.var_names and not self.is_indirect()
        return isinstance(self.index, (int, sympy.Integer))

    def is_indirect(self) -> bool:
        return any(is_indirect(v.name) for v in self.index.free_symbols)  # type: ignore[attr-defined]


@dataclasses.dataclass(frozen=True)
class StarDep(Dep):
    name: str
    mode: Optional[str] = None

    # depends on the entire buffer
    @property
    def index(self) -> sympy.Expr:
        raise NotImplementedError("StarDep does not have an index")

    def get_numel(self) -> sympy.Expr:
        return V.graph.get_numel(self.name)  # type: ignore[return-value]

    def rename(self, renames: dict[str, str]) -> "StarDep":
        if self.name in renames:
            return StarDep(renames[self.name], self.mode)
        return self

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    def numbytes_hint(self) -> int:
        try:
            return V.graph.sizevars.size_hint(self.get_numel()) * get_dtype_size(
                V.graph.get_dtype(self.name)
            )
        except NotImplementedError:
            return 0  # NoneLayout, MultiOutputLayout, etc

    def has_unbacked_symbols(self) -> bool:
        return len(free_unbacked_symbols(self.get_numel())) > 0

    def is_contiguous(self) -> bool:
        return False

    def is_scalar(self) -> bool:
        return False

    def is_indirect(self) -> bool:
        return False


# Used for tracking mutation ordering
# if A reads a buffer and B mutates it
# B must be ordered after A
#
# This is useful for a variety of reasons.
# For example, if A's read is never actually used, we can eliminate it.
# Another case is if A's buffer ends up being fused away, we never need to
# materialize that buffer
@dataclasses.dataclass(frozen=True)
class WeakDep(Dep):
    # Fake dependency on unused buffer
    name: str
    # Buffer that is doing the mutation
    mutating_buf: str
    # WeakDep's are also used to add dependencies to prevent some specific reordering,
    # E.g. collectives global ordering.
    # But if other pass guarantees proper ordering by its logic,
    # This additional "fake" deps will be holding optimizations.
    # This flag is used to identify those additional deps.
    is_fake: bool = False

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return OrderedSet()

    @property
    def index(self) -> sympy.Expr:
        raise NotImplementedError("WeakDep does not have an index")

    def get_numel(self) -> sympy.Expr:
        return sympy.S.One

    def rename(self, renames: dict[str, str]) -> "WeakDep":
        if self.name in renames:
            return WeakDep(renames[self.name], self.mutating_buf, self.is_fake)
        return self

    def numbytes_hint(self) -> int:
        return 1  # Purely inserted for ordering, not an actual dep

    def has_unbacked_symbols(self) -> bool:
        return False

    def is_contiguous(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class IndexExprDep:
    index: sympy.Expr  # type: ignore[assignment]
    var_names: tuple[sympy.Symbol, ...]
    size: tuple[sympy.Expr, ...]


@dataclasses.dataclass
class ReadWrites:
    reads: OrderedSet[Dep]
    writes: OrderedSet[Dep]
    index_exprs: OrderedSet[IndexExprDep]
    range_vars: Optional[list[sympy.Expr]] = None
    var_ranges: Optional[VarRanges] = None

    def rename(self, renames: dict[str, str]) -> "ReadWrites":
        return ReadWrites(
            OrderedSet(dep.rename(renames) for dep in self.reads),
            OrderedSet(dep.rename(renames) for dep in self.writes),
            self.index_exprs,
            self.range_vars,
            self.var_ranges,
        )

    def with_read(self, dep: Union[Dep, OrderedSet[Dep]]) -> "ReadWrites":
        assert isinstance(dep, (WeakDep, StarDep, OrderedSet))
        if not isinstance(dep, OrderedSet):
            dep = OrderedSet([dep])
        return ReadWrites(
            OrderedSet.union(self.reads, dep),
            self.writes,
            self.index_exprs,
            self.range_vars,
            self.var_ranges,
        )

    def merge(self, other: "ReadWrites") -> "ReadWrites":
        reads = OrderedSet.union(self.reads, other.reads)
        writes = OrderedSet.union(self.writes, other.writes)
        index_exprs = OrderedSet.union(self.index_exprs, other.index_exprs)
        return ReadWrites(reads - writes, writes, index_exprs)

    @staticmethod
    def merge_list(read_writes: list["ReadWrites"]) -> "ReadWrites":
        all_writes = OrderedSet.union(*[rw.writes for rw in read_writes])
        all_reads = OrderedSet.union(*[rw.reads for rw in read_writes]) - all_writes
        all_index_exprs = OrderedSet.union(*[rw.index_exprs for rw in read_writes])
        return ReadWrites(all_reads, all_writes, all_index_exprs)

    def remove_reads(self, rem_reads: OrderedSet[Dep]) -> "ReadWrites":
        return ReadWrites(
            self.reads - rem_reads,
            self.writes,
            self.index_exprs,
            self.range_vars,
            self.var_ranges,
        )

    def reads_and_writes(self) -> Iterable[Dep]:
        return itertools.chain(self.reads, self.writes)

    def buffer_names(self, ignore_integer_index: bool = True) -> OrderedSet[str]:
        """
        Integer index is used for load_seed.
        """
        names: OrderedSet[str] = OrderedSet()
        for dep in self.reads_and_writes():
            if not isinstance(dep, MemoryDep):
                continue
            if not ignore_integer_index or not isinstance(
                dep.index, (int, sympy.Integer)
            ):
                names.add(dep.name)
        return names

    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        result: OrderedSet[sympy.Symbol] = OrderedSet()

        for dep in self.reads_and_writes():
            result |= dep.get_free_symbol_uses(unbacked_only)
        return result


class _RecordLoadStoreInner(V.MockHandler):  # type: ignore[name-defined]
    def __init__(self, var_ranges: VarRanges, normalize: bool) -> None:
        super().__init__()
        self._reads: OrderedSet[Dep] = OrderedSet()
        self._writes: OrderedSet[MemoryDep] = OrderedSet()
        self._index_exprs: OrderedSet[IndexExprDep] = OrderedSet()
        self._var_ranges: VarRanges = var_ranges
        self._should_normalize: bool = normalize

    @staticmethod
    def drop_unused_symbols(
        index: Union[int, sympy.Expr],
        var_names: list[sympy.Expr],
        sizes: list[sympy.Expr],
    ) -> None:
        """
        Reduction has last (reduced) dim in its sizes, but
        downstream users won't.  Normalize this away.
        """
        if not isinstance(index, sympy.Expr):
            # index can be an int
            return
        free_symbols = index.free_symbols
        while var_names and var_names[-1] not in free_symbols:
            var_names.pop()
            sizes.pop()

    @classmethod
    def _normalize(
        cls, index: sympy.Expr, var_ranges: VarRanges
    ) -> tuple[sympy.Expr, tuple[sympy.Symbol, ...], tuple[sympy.Expr, ...]]:
        # Try to further simplify the indexes even if simplify_loops didn't
        # convert it to the simplest form because of the interference from
        # different indexing formulas.
        index_vars = [*var_ranges.keys()]
        sizes = tuple(var_ranges.values())  # type: ignore[assignment]
        new_sizes, reindex, _prune = V.graph.sizevars._simplify_loops(
            index_vars,
            sizes,
            index_prevent_reordering([index], index_vars, sizes),
        )

        # assign new variables each dimension to deal with numbering mismatches
        # d0, d1, d2 could become d0, d2 -- which won't match d0, d1
        new_vars, add_var = var_builder(canonicalization_prefix())
        replacement = dict(zip(index_vars, reindex([add_var(x) for x in new_sizes])))
        index = sympy_subs(sympy.expand(index), replacement)

        new_vars = [*new_vars.keys()]
        new_sizes = [*new_sizes]
        cls.drop_unused_symbols(index, new_vars, new_sizes)
        return index, tuple(new_vars), tuple(new_sizes)  # type: ignore[arg-type]

    def canonicalize(
        self, index: sympy.Expr
    ) -> tuple[sympy.Expr, tuple[sympy.Symbol, ...], tuple[sympy.Expr, ...]]:
        if not self._should_normalize:
            sizes = [V.graph.sizevars.simplify(x) for x in self._var_ranges.values()]
            var_names = [k for k, v in zip(self._var_ranges.keys(), sizes) if v != 1]
            sizes = [v for v in sizes if v != 1]

            self.drop_unused_symbols(index, var_names, sizes)

            return index, tuple(var_names), tuple(sizes)  # type: ignore[return-value, arg-type]
        var_ranges = {
            k: V.graph.sizevars.simplify(v)
            for k, v in self._var_ranges.items()
            # TODO(jansel): explore this further normalization
            # if k in free_symbols
        }
        return self._normalize(index, var_ranges)

    def load(self, name: str, index: sympy.Expr) -> str:
        self._reads.add(MemoryDep(name, *self.canonicalize(index)))
        return f"load({name}, {sympy_str(index)})"

    def load_seed(self, name: str, index: int) -> str:
        assert isinstance(index, int)
        return self.load(name, sympy.Integer(index))

    def store(
        self, name: str, index: sympy.Expr, value: str, mode: Optional[str] = None
    ) -> str:
        self._writes.add(MemoryDep(name, *self.canonicalize(index), mode=mode))
        return f"store({name}, {sympy_str(index)}, {value}, {mode})"

    def store_reduction(self, name: str, index: sympy.Expr, value: str) -> str:
        return self.store(name, index, f"store_reduction({value})")

    def index_expr(self, index: sympy.Expr, dtype: Optional[torch.dtype]) -> str:
        self._index_exprs.add(IndexExprDep(*self.canonicalize(index)))
        return f"index_expr({sympy_str(index)}, {dtype})"

    def bucketize(
        self,
        values: T,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: T,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: Optional[tuple[str, sympy.Expr]] = None,
        sorter_indices: Optional[T] = None,
    ) -> None:
        """Records the names of the buffers that bucketize will read from."""
        self._reads.add(StarDep(boundaries[0]))
        if sorter is not None:
            self._reads.add(StarDep(sorter[0]))


class RecordLoadStore(V.KernelFormatterHandler):  # type: ignore[name-defined]
    def __init__(self, var_ranges: VarRanges, normalize: bool) -> None:
        parent_handler = _RecordLoadStoreInner(
            var_ranges=var_ranges, normalize=normalize
        )
        super().__init__(parent_handler=parent_handler)


# TODO: check call sites
def var_builder(prefix: str) -> tuple[VarRanges, Callable[[sympy.Expr], sympy.Symbol]]:
    cnt = itertools.count()
    var_ranges: VarRanges = {}

    def add_var(length: sympy.Expr) -> sympy.Symbol:
        v = sympy_index_symbol(f"{prefix}{next(cnt)}")
        var_ranges[v] = length
        return v

    return var_ranges, add_var


def index_vars_no_squeeze(
    *argsizes: Sequence[sympy.Expr], prefix: str
) -> tuple[list[list[sympy.Symbol]], VarRanges]:
    var_ranges, add_var = var_builder(prefix)
    args: list[list[sympy.Symbol]] = [list(map(add_var, size)) for size in argsizes]
    return args, var_ranges


def index_vars_squeeze(
    *argsizes: Sequence[sympy.Expr], prefix: str = "d"
) -> tuple[list[Sequence[sympy.Expr]], VarRanges]:
    from .ir import SqueezeView

    var_ranges, add_var = var_builder(prefix)
    args: list[Sequence[sympy.Expr]] = []
    new_sizes: list[Sequence[sympy.Expr]] = []
    for size in argsizes:
        new_size, reindex = SqueezeView.squeezer(size)
        new_sizes.append(new_size)
        args.append(reindex(list(map(add_var, new_size))))
    return args, var_ranges


def extract_read_writes(
    fn: Callable[..., Any],
    *argsizes: Sequence[sympy.Expr],
    normalize: bool = False,
    prefix: str = "d",
    hidden_args: Sequence[list[sympy.Expr]] = (),
) -> ReadWrites:
    args, var_ranges = index_vars_squeeze(*argsizes, prefix=prefix)

    from .loop_body import LoopBody

    if isinstance(fn, LoopBody):
        inner = extract_loop_body_with_args(
            fn,
            [*args, *hidden_args],  # type: ignore[list-item]
            var_ranges,
            normalize,
        )
    else:
        # Slow path tracing the function
        rw = RecordLoadStore(var_ranges, normalize=normalize)
        with V.set_ops_handler(rw):
            fn(*args, *hidden_args)
        inner = rw.parent_handler

    if normalize:
        range_vars = []  # Number of vars could differ due to normalization
    else:
        range_vars = [*itertools.chain.from_iterable(args)]

    return ReadWrites(
        OrderedSet(inner._reads),
        OrderedSet(inner._writes),
        inner._index_exprs,
        range_vars,
        var_ranges,
    )


def extract_loop_body_with_args(
    fn: Any,
    args: list[list[sympy.Expr]],
    var_ranges: VarRanges,
    normalize: bool = False,
) -> _RecordLoadStoreInner:
    from .loop_body import MemoryUsageType

    # Fast path to avoid tracing when we already have a LoopBody
    inner = _RecordLoadStoreInner(var_ranges=var_ranges, normalize=normalize)
    name_to_index = fn.indexing_from_args(args)
    if fn.indirect_vars:
        # mimic the `tmpX` naming tracing gives us
        repl = {v: make_symbol(SymT.TMP, i) for i, v in enumerate(fn.indirect_vars)}
        name_to_index = {k: sympy_subs(v, repl) for k, v in name_to_index.items()}  # type: ignore[arg-type]
    for entry in fn.memory_usage[MemoryUsageType.LOAD]:
        inner.load(entry.buffer_name, name_to_index[entry.index_name])  # type: ignore[arg-type]
    for entry in fn.memory_usage[MemoryUsageType.LOAD_SEED]:
        inner.load_seed(entry.buffer_name, int(name_to_index[entry.index_name]))  # type: ignore[arg-type]
    for entry in fn.memory_usage[MemoryUsageType.STORE]:
        inner.store(
            entry.buffer_name,
            name_to_index[entry.index_name],
            None,  # type: ignore[arg-type]
            entry.mode,
        )
    for entry in fn.memory_usage[MemoryUsageType.STORE_REDUCTION]:
        inner.store_reduction(
            entry.buffer_name,
            name_to_index[entry.index_name],
            None,  # type: ignore[arg-type]
        )
    for entry in fn.memory_usage[MemoryUsageType.INDEX_EXPR]:
        inner.index_expr(name_to_index[entry.index_name], None)
    for entry in fn.memory_usage[MemoryUsageType.BUCKETIZE]:
        # All that matters is that we record the buffer name, so place it in the
        # "boundaries" name position to ensure that it's recorded.
        inner.bucketize(
            None,
            (entry.buffer_name, None, None, None),
            None,
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
        )
    # fn.memory_usage[MemoryUsageType.CHECK_BOUNDS] intentionally skipped
    return inner


def extract_input_node_reduction_ranges(
    input_node: "torch._inductor.ir.IRNode",
) -> tuple[Optional[list[sympy.Expr]], Optional[list[sympy.Expr]]]:
    """
    Returns the size and reduction size of all inputs, if the sizes and reduction_sizes (if exist) are all the same.
    It's possible that a node has multiple inputs, some are Reduction nodes and others are Pointwise nodes.
    In this case, reduction_sizes of the Reduction nodes need to be the same.
    Otherwise returns (None, None).
    """

    from .ir import ComputedBuffer, ExternKernel, Loops

    size: Optional[list[sympy.Expr]]
    reduction_size: Optional[list[sympy.Expr]]

    if isinstance(input_node.get_defining_op(), ComputedBuffer):
        # Input node has already been realized. Return its size and reduction_size.
        size = [*input_node.get_size()]
        reduction_size = [*input_node.get_reduction_size()]
        if len(reduction_size) > 0:
            return (size, reduction_size)
        else:
            return (None, None)

    if not isinstance(input_node.data.data, Loops):  # type: ignore[attr-defined]
        # Other IRNodes do not have reduction_ranges.
        return (None, None)

    # There is one issue: what if there are views / permutations between the input node and its dependent realized nodes?
    # The current method still uses reduction ranges from the dependent realized node, which is not ideal.
    # Is there a way to check whether there are permutations in between?
    reads = input_node.get_reads()
    reduction_size: Optional[list[sympy.Expr]] = None
    size: Optional[list[sympy.Expr]] = None
    while reduction_size is None and len(reads) > 0:
        seen: OrderedSet[str] = OrderedSet()
        new_reads: list[Dep] = []
        for read in reads:
            if not isinstance(read, MemoryDep):
                continue
            if read.name in seen:
                continue
            seen.add(read.name)
            buffer = V.graph.try_get_buffer(read.name)
            if buffer is None:
                continue
            op = buffer.get_defining_op()
            if op is None or isinstance(op, ExternKernel):
                continue

            if isinstance(op, ComputedBuffer) and len(op.get_reduction_size()) > 0:
                if reduction_size is None:
                    reduction_size = [*op.get_reduction_size()]
                    size = [*op.get_size()]
                elif reduction_size != [*op.get_reduction_size()] or size != [
                    *op.get_size()
                ]:
                    return (None, None)
            else:
                new_reads.extend(op.get_reads())
        if reads == new_reads:
            return (size, reduction_size)
        else:
            reads = OrderedSet(new_reads)
    return (size, reduction_size)


def canonicalization_prefix() -> str:
    return "c"


# ops handler which computes all the free symbols for an IR
class FreeSymbolsOpsHandler(DefaultHandler):
    symbols: OrderedSet[sympy.Symbol]

    def __init__(self, unbacked_only: bool = True) -> None:
        self.symbols = OrderedSet()
        self.get_symbols = free_unbacked_symbols if unbacked_only else free_symbols

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        for a in itertools.chain(args, kwargs.values()):
            if isinstance(a, (sympy.Expr, sympy.logic.boolalg.Boolean)):
                self.symbols |= self.get_symbols(a)

    def indirect_indexing(
        self,
        index_var: Any,
        size: Union[int, sympy.Expr],
        check: bool = True,
        wrap_neg: bool = True,
    ) -> sympy.Symbol:
        assert not isinstance(index_var, (sympy.Expr, sympy.logic.boolalg.Boolean))
        self.symbols |= self.get_symbols(size)
        return sympy_index_symbol(f"({str(index_var)})")

    def frexp(self, x: Any) -> tuple[None, ...]:
        return (None,) * 2

    def scan(
        self, dtypes: Any, combine_fn: Any, values: Sequence[Any]
    ) -> tuple[None, ...]:
        return (None,) * len(values)

    def sort(
        self, dtypes: Any, values: Sequence[Any], stable: Any, descending: Any
    ) -> tuple[None, ...]:
        return (None,) * len(values)

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[None, tuple[None, ...]],
    ) -> Union[None, tuple[None, ...]]:
        num_values = reduction_num_outputs(reduction_type)
        return (None,) * num_values if num_values > 1 else None

    def masked(self, mask: Any, body: Callable[..., Any], other: Any) -> None:
        assert callable(body), "masked body must always be callable."
        # The body can make additional calls, for e.g. ops.indirect_indexing
        body()


def extract_free_symbols(
    fn: Callable[..., Any],
    index: Sequence[sympy.Expr],
    rindex: Optional[Sequence[sympy.Expr]] = None,
    unbacked_only: bool = True,
) -> OrderedSet[sympy.Symbol]:
    from .ir import FlexibleLayout

    args = [index, rindex] if rindex is not None else [index]
    handler = FreeSymbolsOpsHandler(unbacked_only)
    # NB: I cargo culted the allow_indexing patch here, I don't understand why
    # people do this all over
    with (
        V.set_ops_handler(handler),
        patch.object(FlexibleLayout, "allow_indexing", True),
    ):
        fn(*args)
    return handler.symbols
