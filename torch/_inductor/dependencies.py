import collections
import dataclasses
import itertools
import logging
import re
import typing
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import sympy

import torch

from .codegen.common import index_prevent_reordering
from .utils import get_dtype_size, sympy_str, sympy_subs, sympy_symbol, VarRanges
from .virtualized import V

log = logging.getLogger(__name__)
is_indirect = re.compile(r"indirect|tmp").search
Dep = Union["MemoryDep", "StarDep", "WeakDep"]


class MemoryDep(typing.NamedTuple):
    name: str
    index: sympy.Expr  # type: ignore[assignment]
    var_names: Tuple[sympy.Symbol, ...]
    size: Tuple[sympy.Expr, ...]

    def __repr__(self):
        return f"MemoryDep({self.name!r}, {self.index}, {self.ranges})"

    @property
    def ranges(self) -> Dict[sympy.Symbol, sympy.Expr]:
        """{c0: 128, c1: 512, ...}"""
        return dict(zip(self.var_names, self.size))

    def rename(self, renames: Dict[str, str]) -> "MemoryDep":
        if self.name in renames:
            return MemoryDep(
                renames[self.name], self.index, var_names=self.var_names, size=self.size
            )
        return self

    def numbytes_hint(self):
        if self.is_indirect():
            numel = V.graph.get_numel(self.name)
        else:
            vars = set(self.index.free_symbols)
            numel = sympy.Integer(1)
            for var, size in zip(self.var_names, self.size):
                if var in vars:
                    numel = numel * size
        return V.graph.sizevars.size_hint(numel) * get_dtype_size(
            V.graph.get_dtype(self.name)
        )

    def is_contiguous(self) -> bool:
        return isinstance(self.index, sympy.Symbol) and self.index in self.var_names

    def is_scalar(self) -> bool:
        if isinstance(self.index, sympy.Symbol):
            return self.index not in self.var_names and not self.is_indirect()
        return isinstance(self.index, (int, sympy.Integer))

    def is_indirect(self) -> bool:
        return any(is_indirect(v.name) for v in self.index.free_symbols)


class StarDep(typing.NamedTuple):
    # depends on the entire buffer
    name: str

    @property
    def index(self):
        raise NotImplementedError("StarDep does not have an index")

    def rename(self, renames: Dict[str, str]) -> "StarDep":
        if self.name in renames:
            return StarDep(renames[self.name])
        return self

    def numbytes_hint(self):
        return V.graph.sizevars.size_hint(
            V.graph.get_numel(self.name)
        ) * get_dtype_size(V.graph.get_dtype(self.name))

    def is_contiguous(self) -> bool:
        return False

    def is_scalar(self) -> bool:
        return False

    def is_indirect(self) -> bool:
        return False


# Used for tracking mutation ordering
# if A reads a buffer and B mutates it
# B must be ordered after A
class WeakDep(typing.NamedTuple):
    name: str

    @property
    def index(self):
        raise NotImplementedError("WeakDep does not have an index")

    def rename(self, renames: Dict[str, str]) -> "WeakDep":
        if self.name in renames:
            return WeakDep(renames[self.name])
        return self

    def numbytes_hint(self):
        return 1  # Purely inserted for ordering, not an actual dep

    def is_contiguous(self) -> bool:
        return False


class IndexExprDep(typing.NamedTuple):
    index: sympy.Expr  # type: ignore[assignment]
    var_names: Tuple[sympy.Symbol, ...]
    size: Tuple[sympy.Expr, ...]


@dataclasses.dataclass
class ReadWrites:
    reads: Set[Dep]
    writes: Set[Dep]
    index_exprs: Set[IndexExprDep]
    range_vars: Optional[List[sympy.Expr]] = None
    var_ranges: Optional[VarRanges] = None
    op_counts: typing.Counter[Any] = None  # type: ignore[assignment]

    def rename(self, renames: typing.Dict[str, str]) -> "ReadWrites":
        return ReadWrites(
            {dep.rename(renames) for dep in self.reads},
            {dep.rename(renames) for dep in self.writes},
            self.index_exprs,
            self.range_vars,
            self.var_ranges,
            op_counts=self.op_counts,
        )

    def with_read(self, dep: Dep) -> "ReadWrites":
        assert isinstance(dep, (WeakDep, StarDep))
        return ReadWrites(
            set.union(self.reads, {dep}),
            self.writes,
            self.index_exprs,
            self.range_vars,
            self.var_ranges,
            op_counts=self.op_counts,
        )

    def merge(self, other: "ReadWrites"):
        reads = set.union(self.reads, other.reads)
        writes = set.union(self.writes, other.writes)
        index_exprs = set.union(self.index_exprs, other.index_exprs)
        if self.op_counts is not None:
            op_counts = collections.Counter(self.op_counts)
            op_counts.update(other.op_counts or {})
        else:
            op_counts = other.op_counts
        return ReadWrites(reads - writes, writes, index_exprs, op_counts=op_counts)

    @staticmethod
    def merge_list(read_writes: List["ReadWrites"]):
        all_writes = set.union(*[rw.writes for rw in read_writes])
        all_reads = set.union(*[rw.reads for rw in read_writes]) - all_writes
        all_index_exprs = set.union(*[rw.index_exprs for rw in read_writes])

        op_counts: typing.Counter[Any] = collections.Counter()
        for rw in read_writes:
            if rw.op_counts is not None:
                op_counts.update(rw.op_counts)

        return ReadWrites(all_reads, all_writes, all_index_exprs, op_counts=op_counts)

    def remove_reads(self, rem_reads):
        return ReadWrites(
            self.reads - rem_reads,
            self.writes,
            self.index_exprs,
            self.range_vars,
            self.var_ranges,
            op_counts=self.op_counts,
        )

    def reads_and_writes(self):
        return itertools.chain(self.reads, self.writes)


class _RecordLoadStoreInner(V.MockHandler):
    def __init__(self, var_ranges: VarRanges, normalize: bool):
        super().__init__()
        self._reads: Set[MemoryDep] = set()
        self._writes: Set[MemoryDep] = set()
        self._index_exprs: Set[IndexExprDep] = set()
        self._var_ranges: VarRanges = var_ranges
        self._normalize: bool = normalize

    def canonicalize(
        self, index: sympy.Expr
    ) -> Tuple[sympy.Expr, Tuple[sympy.Expr, ...]]:
        if not self._normalize:
            sizes = [V.graph.sizevars.simplify(x) for x in self._var_ranges.values()]
            var_names = tuple(
                k for k, v in zip(self._var_ranges.keys(), sizes) if v != 1
            )
            sizes = tuple(v for v in sizes if v != 1)
            return index, var_names, sizes  # type: ignore[return-value]

        # Try to further simplify the indexes even if simplify_loops didn't
        # convert it to the simplest form because of the interference from
        # different indexing formulas.
        free_symbols = index.free_symbols
        var_ranges = {
            k: V.graph.sizevars.simplify(v)
            for k, v in self._var_ranges.items()
            # TODO(jansel): explore this further normalization
            # if k in free_symbols
        }
        index_vars = [*var_ranges.keys()]
        sizes = [*var_ranges.values()]  # type: ignore[assignment]
        new_sizes, reindex, prune = V.graph.sizevars._simplify_loops(
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
        free_symbols = index.free_symbols
        while new_vars and new_vars[-1] not in free_symbols:
            # Reduction has last (reduced) dim in its sizes, but
            # downstream users won't.  Normalize this away.
            new_vars.pop()
            new_sizes.pop()
        return index, tuple(new_vars), tuple(new_sizes)  # type: ignore[return-value]

    def load(self, name: str, index: sympy.Expr) -> str:
        self._reads.add(MemoryDep(name, *self.canonicalize(index)))  # type: ignore[call-arg]
        return f"load({name}, {sympy_str(index)})"

    def load_seed(self, name: str, index: int):
        assert isinstance(index, int)
        return self.load(name, sympy.Integer(index))

    def store(self, name: str, index: sympy.Expr, value: str, mode=None) -> str:
        self._writes.add(MemoryDep(name, *self.canonicalize(index)))  # type: ignore[call-arg]
        return f"store({name}, {sympy_str(index)}, {value}, {mode})"

    def store_reduction(self, name: str, index, value) -> str:
        return self.store(name, index, f"store_reduction({value})")

    def index_expr(self, index: sympy.Expr, dtype) -> str:
        self._index_exprs.add(IndexExprDep(*self.canonicalize(index)))  # type: ignore[call-arg]
        return f"index_expr({sympy_str(index)}, {dtype})"

    def bucketize(
        self,
        values,
        offsets_name: str,
        offsets_size: sympy.Expr,
        indexing_dtype: torch.dtype,
        right: bool,
    ):
        self._reads.add(StarDep(offsets_name))  # type: ignore[arg-type]
        return f"bucketize({values}, {offsets_name}, {sympy_str(offsets_size)}, {indexing_dtype}, {right})"


class _OpCounter:
    """Shim to count how many times each op is used"""

    def __init__(self, inner):
        super().__init__()
        self.parent_handler = inner
        self._op_counts: typing.Counter[Any] = collections.Counter()

    def __getattr__(self, name):
        self._op_counts[name] += 1
        return getattr(self.parent_handler, name)


class RecordLoadStore(V.KernelFormatterHandler):
    def __init__(self, var_ranges: VarRanges, normalize: bool):
        parent_handler = _RecordLoadStoreInner(
            var_ranges=var_ranges, normalize=normalize
        )
        parent_handler = _OpCounter(parent_handler)
        super().__init__(parent_handler=parent_handler)


def var_builder(prefix: str) -> Tuple[VarRanges, Callable[[sympy.Expr], sympy.Symbol]]:
    cnt = itertools.count()
    var_ranges: VarRanges = dict()

    def add_var(length: sympy.Expr) -> sympy.Symbol:
        v = sympy_symbol(f"{prefix}{next(cnt)}")
        var_ranges[v] = length
        return v

    return var_ranges, add_var


def index_vars_no_squeeze(*argsizes: Tuple[sympy.Expr, ...], prefix: str):
    var_ranges, add_var = var_builder(prefix)
    args: List[List[sympy.Symbol]] = []
    for size in argsizes:
        args.append(list(map(add_var, size)))
    return args, var_ranges


def index_vars_squeeze(*argsizes: Tuple[sympy.Expr, ...], prefix: str = "d"):
    from .ir import SqueezeView

    var_ranges, add_var = var_builder(prefix)
    args: List[List[sympy.Expr]] = []
    new_sizes: List[List[sympy.Expr]] = []
    for size in argsizes:
        new_size, reindex = SqueezeView.squeezer(size)
        new_sizes.append(new_size)
        args.append(reindex(list(map(add_var, new_size))))
    return args, var_ranges


def extract_read_writes(
    fn: Callable[[sympy.Expr], Any],
    *argsizes: Tuple[sympy.Expr, ...],
    normalize: bool = False,
    prefix: str = "d",
):
    args, var_ranges = index_vars_squeeze(*argsizes, prefix=prefix)
    rw = RecordLoadStore(var_ranges, normalize=normalize)
    with V.set_ops_handler(rw):  # type: ignore[call-arg]
        fn(*args)

    if normalize:
        range_vars = []  # Number of vars could differ due to normalization
    else:
        range_vars = [*itertools.chain(*args)]

    inner = rw.parent_handler.parent_handler
    return ReadWrites(
        set(inner._reads),
        set(inner._writes),
        inner._index_exprs,
        range_vars,
        var_ranges,
        rw.parent_handler._op_counts,
    )


def canonicalization_prefix():
    return "c"
