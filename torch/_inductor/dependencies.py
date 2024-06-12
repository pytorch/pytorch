import abc
import collections
import dataclasses
import itertools
import logging
import re
import typing
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import patch

import sympy

import torch
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

from .codegen.common import index_prevent_reordering
from .utils import (
    get_dtype_size,
    reduction_num_outputs,
    sympy_index_symbol,
    sympy_str,
    sympy_subs,
    VarRanges,
)
from .virtualized import OpsHandler, ReductionType, V

log = logging.getLogger(__name__)
is_indirect = re.compile(r"indirect|tmp").search


class Dep(abc.ABC):
    name: str
    index: sympy.Expr

    @abc.abstractmethod
    def rename(self, renames: Dict[str, str]) -> "Dep":
        pass

    @abc.abstractmethod
    def get_numel(self) -> sympy.Expr:
        pass

    @abc.abstractmethod
    def numbytes_hint(self):
        pass

    @abc.abstractmethod
    def has_unbacked_symbols(self) -> bool:
        pass

    @abc.abstractmethod
    def is_contiguous(self) -> bool:
        pass


@dataclasses.dataclass(frozen=True)
class MemoryDep(Dep):
    name: str
    index: sympy.Expr
    var_names: Tuple[sympy.Symbol, ...]
    size: Tuple[sympy.Expr, ...]
    mode: Optional[str] = None

    def __repr__(self):
        return f"MemoryDep({self.name!r}, {self.index}, {self.ranges}, {self.mode})"

    def get_offset(self):
        """
        Return the offset by setting every variable to be 0.
        """
        return sympy_subs(self.index, {v: 0 for v in self.var_names})

    def normalize_with_stride_order(self, prefix="t"):
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

        new_simplified_sizes, reindex, prune = V.graph.sizevars._simplify_loops(
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
        new_index = sympy_subs(sympy.expand(self.index), replacement)

        out = MemoryDep(self.name, new_index, tuple(var_ranges.keys()), tuple(var_ranges.values()))  # type: ignore[arg-type]
        return out

    @property
    def ranges(self) -> Dict[sympy.Symbol, sympy.Expr]:
        """{c0: 128, c1: 512, ...}"""
        return dict(zip(self.var_names, self.size))

    def get_numel(self) -> sympy.Expr:
        if self.is_indirect():
            numel = V.graph.get_numel(self.name)
        else:
            vars = set(self.index.free_symbols)
            numel = sympy.Integer(1)
            for var, size in zip(self.var_names, self.size):
                if var in vars:
                    numel = numel * size
        return numel

    def rename(self, renames: Dict[str, str]) -> "MemoryDep":
        if self.name in renames:
            return MemoryDep(
                renames[self.name],
                self.index,
                var_names=self.var_names,
                size=self.size,
                mode=self.mode,
            )
        return self

    def numbytes_hint(self):
        return V.graph.sizevars.size_hint(self.get_numel()) * get_dtype_size(
            V.graph.get_dtype(self.name)
        )

    def has_unbacked_symbols(self):
        return len(free_unbacked_symbols(self.get_numel())) > 0

    def is_contiguous(self) -> bool:
        return isinstance(self.index, sympy.Symbol) and self.index in self.var_names

    def stride1_for_last_dim(self, result_for_complex_expression=True) -> bool:
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
            if term is last_sym:
                return True

            # Having a >1 stride for the last dimension is bad for perf
            # return False.
            if (
                isinstance(term, sympy.Mul)
                and len(term.args) == 2
                and term.args[1] is last_sym
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
    def index(self):
        raise NotImplementedError("StarDep does not have an index")

    def get_numel(self) -> sympy.Expr:
        return V.graph.get_numel(self.name)

    def rename(self, renames: Dict[str, str]) -> "StarDep":
        if self.name in renames:
            return StarDep(renames[self.name], self.mode)
        return self

    def numbytes_hint(self):
        return V.graph.sizevars.size_hint(self.get_numel()) * get_dtype_size(
            V.graph.get_dtype(self.name)
        )

    def has_unbacked_symbols(self):
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
    name: str

    @property
    def index(self):
        raise NotImplementedError("WeakDep does not have an index")

    def get_numel(self) -> sympy.Expr:
        return sympy.Integer(1)

    def rename(self, renames: Dict[str, str]) -> "WeakDep":
        if self.name in renames:
            return WeakDep(renames[self.name])
        return self

    def numbytes_hint(self):
        return 1  # Purely inserted for ordering, not an actual dep

    def has_unbacked_symbols(self):
        return False

    def is_contiguous(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class IndexExprDep:
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
    op_counts: typing.Counter[str] = dataclasses.field(
        default_factory=collections.Counter
    )

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
        op_counts = collections.Counter(self.op_counts)
        op_counts.update(other.op_counts)
        return ReadWrites(reads - writes, writes, index_exprs, op_counts=op_counts)

    @staticmethod
    def merge_list(read_writes: List["ReadWrites"]):
        all_writes = set.union(*[rw.writes for rw in read_writes])
        all_reads = set.union(*[rw.reads for rw in read_writes]) - all_writes
        all_index_exprs = set.union(*[rw.index_exprs for rw in read_writes])

        op_counts: typing.Counter[Any] = collections.Counter()
        for rw in read_writes:
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

    def buffer_names(self, ignore_integer_index=True):
        """
        Integer index is used for load_seed.
        """
        names = set()
        for dep in self.reads_and_writes():
            if not isinstance(dep, MemoryDep):
                continue
            if not ignore_integer_index or not isinstance(
                dep.index, (int, sympy.Integer)
            ):
                names.add(dep.name)
        return names


class _RecordLoadStoreInner(V.MockHandler):  # type: ignore[name-defined]
    def __init__(self, var_ranges: VarRanges, normalize: bool):
        super().__init__()
        self._reads: Set[Dep] = set()
        self._writes: Set[MemoryDep] = set()
        self._index_exprs: Set[IndexExprDep] = set()
        self._var_ranges: VarRanges = var_ranges
        self._normalize: bool = normalize

    def canonicalize(
        self, index: sympy.Expr
    ) -> Tuple[sympy.Expr, Tuple[sympy.Symbol, ...], Tuple[sympy.Expr, ...]]:
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
        sizes = tuple(var_ranges.values())
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
        return index, tuple(new_vars), tuple(new_sizes)  # type: ignore[arg-type]

    def load(self, name: str, index: sympy.Expr) -> str:
        self._reads.add(MemoryDep(name, *self.canonicalize(index)))
        return f"load({name}, {sympy_str(index)})"

    def load_seed(self, name: str, index: int):
        assert isinstance(index, int)
        return self.load(name, sympy.Integer(index))

    def store(self, name: str, index: sympy.Expr, value: str, mode=None) -> str:
        self._writes.add(MemoryDep(name, *self.canonicalize(index), mode=mode))
        return f"store({name}, {sympy_str(index)}, {value}, {mode})"

    def store_reduction(self, name: str, index, value) -> str:
        return self.store(name, index, f"store_reduction({value})")

    def index_expr(self, index: sympy.Expr, dtype) -> str:
        self._index_exprs.add(IndexExprDep(*self.canonicalize(index)))
        return f"index_expr({sympy_str(index)}, {dtype})"

    def bucketize(
        self,
        values,
        offsets_name: str,
        offsets_size: sympy.Expr,
        indexing_dtype: torch.dtype,
        right: bool,
    ):
        self._reads.add(StarDep(offsets_name))
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


class RecordLoadStore(V.KernelFormatterHandler):  # type: ignore[name-defined]
    def __init__(self, var_ranges: VarRanges, normalize: bool):
        parent_handler = _RecordLoadStoreInner(
            var_ranges=var_ranges, normalize=normalize
        )
        parent_handler = _OpCounter(parent_handler)
        super().__init__(parent_handler=parent_handler)


# TODO: check call sites
def var_builder(prefix: str) -> Tuple[VarRanges, Callable[[sympy.Expr], sympy.Symbol]]:
    cnt = itertools.count()
    var_ranges: VarRanges = dict()

    def add_var(length: sympy.Expr) -> sympy.Symbol:
        v = sympy_index_symbol(f"{prefix}{next(cnt)}")
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
    fn: Callable[..., Any],
    *argsizes: Tuple[sympy.Expr, ...],
    normalize: bool = False,
    prefix: str = "d",
):
    args, var_ranges = index_vars_squeeze(*argsizes, prefix=prefix)
    rw = RecordLoadStore(var_ranges, normalize=normalize)
    with V.set_ops_handler(rw):
        fn(*args)

    if normalize:
        range_vars = []  # Number of vars could differ due to normalization
    else:
        range_vars = list(itertools.chain.from_iterable(args))

    inner = rw.parent_handler.parent_handler
    return ReadWrites(
        set(inner._reads),
        set(inner._writes),
        inner._index_exprs,
        range_vars,
        var_ranges,
        rw.parent_handler._op_counts,
    )


def extract_input_node_reduction_ranges(
    input_node: "torch._inductor.ir.TensorBox",
) -> Tuple[Optional[List[sympy.Expr]], Optional[List[sympy.Expr]]]:
    """
    Returns the size and reduction size of all inputs, if the sizes and reduction_sizes (if exist) are all the same.
    It's possible that a node has multiple inputs, some are Reduction nodes and others are Pointwise nodes.
    In this case, reduction_sizes of the Reduction nodes need to be the same.
    Otherwise returns (None, None).
    """

    from .ir import ComputedBuffer, Loops

    if isinstance(input_node.data, ComputedBuffer):
        # Input node has already been realized. Return its size and reduction_size.
        size = input_node.get_size()
        reduction_size = input_node.get_reduction_size()
        if len(reduction_size) > 0:
            return (size, reduction_size)
        else:
            return (None, None)

    if not isinstance(input_node.data.data, Loops):  # type: ignore[attr-defined]
        # Other IRNodes do not have reduction_ranges.
        return (None, None)

    # There is one issue: what if there are views / permutations between the input node and its dependent realized nodes?
    # The current method still uses reduction ranges from the dependent realized node, which is not ideal.
    # Is there a way to check whether there are permutations inbetween?
    reads = input_node.get_reads()
    reduction_size = None
    size = None
    while reduction_size is None and len(reads) > 0:
        seen = set()
        new_reads = []
        for read in reads:
            if not isinstance(read, MemoryDep):
                continue
            if read.name in seen:
                continue
            seen.add(read.name)
            buffer = V.graph.get_buffer(read.name)
            if buffer is None:
                continue
            if (
                isinstance(buffer, ComputedBuffer)
                and len(buffer.get_reduction_size()) > 0
            ):
                if reduction_size is None:
                    reduction_size = buffer.get_reduction_size()
                    size = buffer.get_size()
                elif (
                    reduction_size != buffer.get_reduction_size()
                    or size != buffer.get_size()
                ):
                    return (None, None)
            else:
                new_reads.extend(buffer.get_reads())
        if reads == new_reads:
            return (size, reduction_size)
        else:
            reads = new_reads
    return (size, reduction_size)


def canonicalization_prefix():
    return "c"


# ops handler which computes all the free unbacked symbols for an IR
class FreeUnbackedSymbolsOpsHandler:
    symbols: Set[sympy.Symbol]

    def __init__(self):
        self.symbols = set()

    def __getattr__(self, name: str) -> Callable[..., Any]:
        def inner(*args, **kwargs):
            for a in itertools.chain(args, kwargs.values()):
                if isinstance(a, (sympy.Expr, sympy.logic.boolalg.Boolean)):
                    self.symbols |= free_unbacked_symbols(a)

        return inner

    def indirect_indexing(self, index_var, size, check=True) -> sympy.Symbol:
        assert not isinstance(index_var, (sympy.Expr, sympy.logic.boolalg.Boolean))
        self.symbols |= free_unbacked_symbols(size)
        return sympy_index_symbol(f"({str(index_var)})")

    def frexp(self, x):
        return (None,) * 2

    def scan(self, dtypes, combine_fn, values):
        return (None,) * len(values)

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[None, Tuple[None, ...]],
    ) -> Union[None, Tuple[None, ...]]:
        num_values = reduction_num_outputs(reduction_type)
        return (None,) * num_values if num_values > 1 else None


def _typecheck_FreeUnbackedSymbolsOpsHandler(
    h: FreeUnbackedSymbolsOpsHandler,
) -> OpsHandler[None]:
    return h


def extract_free_unbacked_symbols(fn: Callable[..., Any], index, rindex=None):
    from .ir import FlexibleLayout

    args = [index, rindex] if rindex is not None else [index]
    handler = FreeUnbackedSymbolsOpsHandler()
    # NB: I cargo culted the allow_indexing patch here, I don't understand why
    # people do this all over
    with V.set_ops_handler(handler), patch.object(
        FlexibleLayout, "allow_indexing", True
    ):
        fn(*args)
    return handler.symbols
