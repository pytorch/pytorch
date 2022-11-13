import collections
import dataclasses
import itertools
import logging
import typing
from typing import Callable, cast, Dict, List, Optional, Set, Tuple, Union

import sympy

from . import config
from .codegen.common import index_prevent_reordering
from .utils import sympy_product, sympy_subs, sympy_symbol, VarRanges
from .virtualized import V

log = logging.getLogger(__name__)

Dep = Union["MemoryDep", "StarDep"]


class MemoryDep(typing.NamedTuple):
    name: str
    index: sympy.Expr  # type: ignore[assignment]
    size: Tuple[sympy.Expr, ...]

    def broadcast_extend_sizes(self, extra_sizes: List[sympy.Expr]) -> "MemoryDep":
        size = (*self.size, *[x for x in extra_sizes if x != 1])
        return MemoryDep(self.name, self.index, size)

    def maybe_swap_sizes(self) -> "MemoryDep":
        # swap only in simple cases where index is trivial and
        # there are just 2 sizes
        if (
            len(self.size) == 2
            and len(self.index.args) == 0
            and cast(sympy.Symbol, self.index).name == canonicalization_prefix() + "0"
        ):
            c = canonicalization_prefix()
            size = (self.size[1], self.size[0])
            s0 = sympy_symbol(c + "0")
            s1 = sympy_symbol(c + "1")
            index = sympy_subs(self.index, {s0: s1})
            return MemoryDep(self.name, index, size)
        else:
            return self

    def strip_last_size(self) -> "MemoryDep":
        nsizes = len(self.size)
        if not (nsizes >= 1 and len(self.index.args) <= nsizes - 1):
            return self
        # make sure last dim index is not used
        prefix = canonicalization_prefix()
        len_prefix = len(prefix)
        prefixes = [
            fs.name[:len_prefix]
            for fs in cast(Set[sympy.Symbol], self.index.free_symbols)
        ]
        assert (
            len(prefixes) == 0 or prefix in prefixes
        ), "index expression should contain canonicalized symbols"
        last_index = f"{prefix}{len(self.size)-1}"
        if last_index not in self.index.free_symbols:
            size = self.size[:-1]
            return MemoryDep(self.name, self.index, size)
        else:
            return self

    def rename(self, renames: Dict[str, str]) -> "MemoryDep":
        if self.name in renames:
            return MemoryDep(renames[self.name], self.index, self.size)
        return self

    def numel_hint(self):
        vars = set(self.index.free_symbols)
        return V.graph.sizevars.size_hint(
            sympy_product([s for s in self.size if s in vars])
        )

    def is_contiguous(self) -> bool:
        return isinstance(self.index, (sympy.Symbol, sympy.Integer))


class StarDep(typing.NamedTuple):
    # depends on the entire buffer
    name: str

    def rename(self, renames: Dict[str, str]) -> "StarDep":
        if self.name in renames:
            return StarDep(renames[self.name])
        return self

    def numel_hint(self):
        return 1

    def is_contiguous(self) -> bool:
        return False


class IndexExprDep(typing.NamedTuple):
    index: sympy.Expr  # type: ignore[assignment]
    size: Tuple[sympy.Expr, ...]


@dataclasses.dataclass
class ReadWrites:
    reads: Set[Dep]
    writes: Set[Dep]
    index_exprs: Set[IndexExprDep]
    range_vars: Optional[List[sympy.Expr]] = None
    var_ranges: Optional[VarRanges] = None

    def rename(self, renames: typing.Dict[str, str]) -> "ReadWrites":
        return ReadWrites(
            {dep.rename(renames) for dep in self.reads},
            {dep.rename(renames) for dep in self.writes},
            self.index_exprs,
            self.range_vars,
            self.var_ranges,
        )

    def with_read(self, name: str) -> "ReadWrites":
        assert isinstance(name, str)
        return ReadWrites(
            set.union(self.reads, {StarDep(name)}),
            self.writes,
            self.index_exprs,
            self.range_vars,
            self.var_ranges,
        )

    def merge(self, other):
        reads = set.union(self.reads, other.reads)
        writes = set.union(self.writes, other.writes)
        index_exprs = set.union(self.index_exprs, other.index_exprs)
        return ReadWrites(
            reads - writes,
            writes,
            index_exprs,
        )


class RecordLoadStore(V.KernelBuilder):  # type: ignore[name-defined]
    def __init__(self, var_ranges: VarRanges, normalize: bool):
        super(RecordLoadStore, self).__init__()
        self._reads: Set[MemoryDep] = set()
        self._writes: Set[MemoryDep] = set()
        self._index_exprs: Set[IndexExprDep] = set()
        self._var_ranges: VarRanges = var_ranges
        self._normalize: bool = normalize

    def canonicalize(
        self, index: sympy.Expr
    ) -> Tuple[sympy.Expr, Tuple[sympy.Expr, ...]]:
        sizes = list(self._var_ranges.values())
        sizes = [V.graph.sizevars.simplify(x) for x in sizes]
        if not self._normalize:
            return index, tuple([x for x in sizes if x != 1])

        # Try to further simplify the indexes even if simplify_loops didn't
        # convert it to the simpliest form because of the interference from
        # different indexing formulas.
        index_vars = list(self._var_ranges.keys())
        new_sizes, reindex, prune = V.graph.sizevars._simplify_loops(
            index_vars,
            sizes,
            index_prevent_reordering([index], index_vars, sizes),
        )

        # assign new variables each dimension to deal with numbering mismatches
        # d0, d1, d2 could become d0, d2 -- which won't match d0, d1
        _, add_var = var_builder(canonicalization_prefix())
        replacement = dict(zip(index_vars, reindex([add_var(x) for x in new_sizes])))

        index = sympy_subs(sympy.expand(index), replacement)
        return index, tuple(new_sizes)

    def load(self, name: str, index: sympy.Expr) -> str:
        canonicalized_index, canonicalized_size = self.canonicalize(index)
        self._reads.add(MemoryDep(name, canonicalized_index, canonicalized_size))
        return self._Expr(format_string="load({}, {})", inputs=[name, index])

    def store(self, name: str, index: sympy.Expr, value: str, mode=None) -> str:
        canonicalized_index, canonicalized_size = self.canonicalize(index)
        self._writes.add(MemoryDep(name, canonicalized_index, canonicalized_size))
        return self._Expr(
            format_string="store({}, {}, {}, {})", inputs=[name, index, value, mode]
        )

    def reduction(
        self, name: str, dtype, src_dtype, reduction_type, index, value
    ) -> str:
        r0 = self._Expr(format_string=f"reduce_{reduction_type})({{}})", inputs=[value])
        return self.store(name, index, r0)

    def index_expr(self, index: sympy.Expr, dtype) -> str:
        canonicalized_index, canonicalized_size = self.canonicalize(index)
        self._index_exprs.add(IndexExprDep(canonicalized_index, canonicalized_size))
        return self._Expr(format_string="index_expr({}, {})", inputs=[index, dtype])


def var_builder(prefix: str) -> Tuple[VarRanges, Callable[[sympy.Expr], sympy.Symbol]]:
    cnt = itertools.count()
    var_ranges: VarRanges = collections.OrderedDict()

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
    return new_sizes, args, var_ranges


def extract_read_writes(
    fn: Callable,
    *argsizes: Tuple[sympy.Expr, ...],
    normalize: bool = False,
    prefix: str = "d",
):
    _, args, var_ranges = index_vars_squeeze(*argsizes, prefix=prefix)
    rw = RecordLoadStore(var_ranges, normalize=normalize)
    with V.set_ops_handler(rw):  # type: ignore[call-arg]
        fn(*args)

    if normalize:
        range_vars = []  # Number of vars could differ due to normalization
    else:
        range_vars = [*itertools.chain(*args)]

    return ReadWrites(
        set(rw._reads), set(rw._writes), rw._index_exprs, range_vars, var_ranges
    )


def canonicalization_prefix():
    return "c"
