# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
from typing import (
    Any,
    Callable,
    Counter,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import sympy

import torch
import torch._logging
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, Identity, ModularIndexing
from torch.utils._sympy.symbol import free_symbol_is_type, symbol_is_type, SymT

from ..._dynamo.utils import counters
from .. import config, ir, scheduler
from ..codecache import code_hash
from ..dependencies import Dep, MemoryDep, StarDep, WeakDep
from ..ir import IRNode, TritonTemplateBuffer
from ..optimize_indexing import indexing_dtype_strength_reduction
from ..runtime.hints import ReductionHint
from ..runtime.runtime_utils import green_text, yellow_text
from ..scheduler import BaseSchedulerNode, BaseScheduling, WhyNoFuse
from ..utils import (
    get_dtype_size,
    IndentedBuffer,
    Placeholder,
    sympy_index_symbol,
    sympy_product,
    sympy_subs,
    unique,
)
from ..virtualized import ops, OpsWrapper, V
from .common import CSEVariable, index_prevent_reordering, Kernel, PythonPrinter
from .multi_kernel import MultiKernel


log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")
schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")
fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")


pexpr = PythonPrinter().doprint


@dataclasses.dataclass
class IterationRanges:
    """
    Each range tree represents multiple sets of iteration indexing
    in a single tiled dimension in the output kernel.

    If you have two loops ranges one (4, 3, 2) and another (4, 6),
    then the range tree will be:
            4 (i0)
        3 (i1)  6 (i3)
        2 (i2)
    Where i0 is shared between both loops, but then the split into
    different indexing vars.  All loop ranges must iterate over
    the same number of elements.
    """

    def __init__(
        self,
        name: str,
        var_list: List[sympy.Symbol],
        var_ranges: Dict[sympy.Symbol, sympy.Expr],
        numel: sympy.Expr,
        prefix: str,
        *,
        kernel: SIMDKernel,
        divisor=sympy.Integer(1),
        length=sympy.Integer(1),
        root: IterationRangesRoot,
    ) -> None:
        super().__init__()
        self.name = name
        self.var_list = var_list
        self.var_ranges = var_ranges
        self.numel = numel
        self.prefix = prefix
        self.divisor = divisor
        self.length = length
        self.kernel = kernel
        self.root = root

    def symbol(self):
        return sympy_index_symbol(self.name)


class IterationRangesRoot(IterationRanges):
    def __init__(
        self,
        name: str,
        numel: sympy.Expr,
        # TODO: this is probably SymTy.INDEX and SymTy.RINDEX
        prefix: str,
        index: int,
        kernel: SIMDKernel,
        pid_cache=None,
        *,
        is_loop: bool,
        tensor_dim: Optional[int],
        grid_dim: Optional[int],
        has_zdim: bool,
    ) -> None:
        if pid_cache is None:
            pid_cache = {}
        super().__init__(
            name=name,
            var_list=[],
            var_ranges={},
            numel=numel,
            prefix=prefix,
            kernel=kernel,
            root=self,
        )
        self.index = index
        # Store all the nodes in one flat list
        self.nodes: Dict[sympy.Expr, IterationRangesEntry] = {}
        # This is for re-ordering program ID in triton mm template
        # pid_cache["tl.program_id(0)"] = pid_m
        self.pid_cache: Dict[str, str] = pid_cache

        # True if the dimension is implemented as a single program looping over
        # the full dimension (currently only used for non-persistent reduction)
        assert not is_loop or (prefix == "r" and grid_dim is None)
        self.is_loop = is_loop
        # Index of corresponding dimension on triton tensors
        self.tensor_dim = tensor_dim
        # Index of corresponding dimension in the triton grid
        self.grid_dim = grid_dim
        self.has_zdim = has_zdim

    def __repr__(self) -> str:
        return f"IterationRangesRoot({self.name!r}, {self.numel}, ...)"

    def cache_clear(self):
        for node in self.nodes.values():
            node.cache_clear()

    def index_sym(self):
        return sympy_index_symbol(f"{self.prefix}index")

    def lookup(self, divisor, length):
        """
        Lookup a given RangeTreeEntry, creating it if needed
        """
        if V.graph.sizevars.statically_known_equals(divisor * length, self.numel):
            expr = FloorDiv(self.index_sym(), divisor)
        else:
            expr = ModularIndexing(self.index_sym(), divisor, length)

        if expr not in self.nodes:
            node = IterationRangesEntry(
                f"{self.prefix}{next(V.kernel.iter_vars_count)}",
                divisor,
                length,
                expr,
                self,
            )
            V.kernel.range_tree_nodes[node.symbol()] = node
            self.var_list.append(node.symbol())
            self.var_ranges[node.symbol()] = length
            self.nodes[expr] = node
        return self.nodes[expr]

    def construct_entries(self, lengths: List[sympy.Expr]):
        divisor = sympy.Integer(1)
        itervars = []
        for length in reversed(lengths):
            itervars.append(self.lookup(divisor, length))
            divisor = divisor * length
        return list(reversed(itervars))

    def construct(self, lengths: List[sympy.Expr]):
        return [e.symbol() for e in self.construct_entries(lengths)]

    def vars_and_sizes(self, index: sympy.Expr):
        """Figure out vars from this tree used in index"""
        nodes = [V.kernel.range_tree_nodes.get(s) for s in index.free_symbols]
        nodes = [n for n in nodes if n and n.prefix == self.prefix]
        nodes.sort(
            key=lambda x: V.graph.sizevars.size_hint(
                x.divisor, fallback=config.unbacked_symint_fallback
            )
        )
        divisor = sympy.Integer(1)
        index_vars = []
        sizes = []

        def add(node):
            nonlocal divisor
            index_vars.append(node.symbol())
            sizes.append(node.length)
            divisor = divisor * node.length

        for node in nodes:
            if not V.graph.sizevars.statically_known_equals(node.divisor, divisor):
                # fill in unused index var
                add(self.lookup(divisor, FloorDiv(node.divisor, divisor)))
                divisor = node.divisor
            add(node)
        if not V.graph.sizevars.statically_known_equals(self.numel, divisor):
            # fill in unused index var
            add(self.lookup(divisor, FloorDiv(self.numel, divisor)))

        return list(reversed(index_vars)), list(reversed(sizes))


class IterationRangesEntry(IterationRanges):
    def __init__(
        self,
        name: str,
        divisor: sympy.Expr,
        length: sympy.Expr,
        expr: sympy.Expr,
        parent: IterationRanges,
    ) -> None:
        super().__init__(
            name=name,
            numel=parent.numel / length,
            var_list=parent.var_list,
            var_ranges=parent.var_ranges,
            prefix=parent.prefix,
            divisor=divisor,
            length=length,
            kernel=parent.kernel,
            root=parent.root,
        )
        self.parent = parent
        self.codegen = functools.lru_cache(None)(self._codegen)
        self.expr = expr

    def __repr__(self) -> str:
        return f"IterationRangesEntry({self.name}, {self.divisor}, {self.length}, {self.expr}, {self.var_ranges})"

    def set_name(self, name):
        self.codegen = lambda: name  # type: ignore[assignment]
        self.codegen.cache_clear = lambda: None  # type: ignore[method-assign]
        self.name = name

    def cache_clear(self):
        self.codegen.cache_clear()

    def _codegen(self):
        V.kernel.codegen_iteration_ranges_entry(self)
        return self.name

    def precomputed_args(self):
        # for dynamic shapes, find parts of indexing expressions that have to be precomputed
        precomputed_args: List[sympy.Expr] = []
        if isinstance(self.expr, sympy.Symbol):
            return precomputed_args
        assert isinstance(self.expr, (FloorDiv, ModularIndexing)), type(self.expr)
        for arg in self.expr.args[1:]:
            if not isinstance(arg, (sympy.Integer, sympy.Symbol)):
                symbols = arg.free_symbols
                if len(symbols) > 0 and all(
                    symbol_is_type(s, SymT.SIZE) for s in symbols
                ):
                    precomputed_args.append(arg)
        return precomputed_args

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


def constant_repr(value):
    if value == float("inf"):
        return 'float("inf")'
    elif value == float("-inf"):
        return 'float("-inf")'
    elif math.isnan(value):
        return 'float("nan")'
    return repr(value)


class SIMDKernel(Kernel):
    """
    Common base class for Triton/Halide codegen which both use flattened indexing rather than loop nests.
    """

    sexpr = pexpr
    kexpr: Callable[[sympy.Expr], str]
    allow_block_ptr = False

    def __init__(
        self,
        *groups,
        index_dtype: str,
        mutations: Optional[OrderedSet[str]] = None,
        pid_cache=None,
        reduction_hint=ReductionHint.DEFAULT,
        override_persistent_reduction=None,
    ) -> None:
        if pid_cache is None:
            pid_cache = {}
        super().__init__()
        self.body = IndentedBuffer()
        self.indexing_code = IndentedBuffer()
        self.numels = [V.graph.sizevars.simplify(s) for s in groups]
        self.mutations: OrderedSet[str] = (
            mutations if mutations is not None else OrderedSet()
        )
        self.range_trees: List[IterationRangesRoot] = []
        self.range_tree_nodes: Dict[sympy.Symbol, IterationRangesEntry] = {}
        self.iter_vars_count = itertools.count()
        self.inside_reduction = self.numels[-1] != 1
        self.reduction_hint = reduction_hint
        self.index_dtype: str = index_dtype
        self.last_usage: OrderedSet[str] = OrderedSet()
        self.buf_accesses: DefaultDict[str, List[Dep]] = collections.defaultdict(list)
        self.persistent_reduction: bool = (
            override_persistent_reduction
            if override_persistent_reduction is not None
            else self.should_use_persistent_reduction()
        )
        self.no_x_dim = self.want_no_x_dim()
        self.code_hash: Union[str, None] = None

        # define this in a closure to make cache local to object
        @functools.lru_cache(None)
        def simplify_indexing(index: sympy.Expr):
            index = V.graph.sizevars.simplify_with_ranges(index, self.var_ranges())
            for tree in self.range_trees:
                index = self.combine_contiguous_dims(index, tree)

            return self.combine_modular_indexing_pairs(index)

        self.simplify_indexing = simplify_indexing
        self.initialize_range_tree(pid_cache)

    def want_no_x_dim(self):
        return False

    def initialize_range_tree(self, pid_cache):
        no_r_dim = not self.inside_reduction or self.numels[-1] == 1

        prefixes = "zyxr"
        active_prefixes = prefixes[-len(self.numels) :]

        grid_dims = "xyz"
        if self.no_x_dim:
            tensor_dims = "r"
        elif no_r_dim:
            tensor_dims = "xyz"
        else:
            tensor_dims = "xyzr"

        tensor_dims = "".join(p for p in tensor_dims if p in active_prefixes)

        for i, prefix in enumerate(active_prefixes):
            is_reduction = prefix == "r"
            tensor_dim = tensor_dims.find(prefix) if prefix in tensor_dims else None
            grid_dim = None if is_reduction else grid_dims.find(prefix)
            index = i if grid_dim is None else grid_dim
            self.range_trees.append(
                IterationRangesRoot(
                    f"{prefix}index",
                    self.numels[i],
                    prefix,
                    index,
                    self,
                    pid_cache=pid_cache,
                    is_loop=is_reduction and not self.persistent_reduction,
                    tensor_dim=tensor_dim,
                    grid_dim=grid_dim,
                    has_zdim="z" in active_prefixes,
                )
            )

    def finalize_indexing(self, indices: Sequence[sympy.Expr]):
        """
        Hook called right before codegen with every index that will be
        used in the fused kernel.
        """

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable):
        prior = self.inside_reduction
        self.inside_reduction = False
        try:
            return self.store(name, index, value)
        finally:
            self.inside_reduction = prior

    def should_use_persistent_reduction(self) -> bool:
        return False  # defined in subclass

    def var_ranges(self):
        return dict(
            itertools.chain.from_iterable(
                tree.var_ranges.items() for tree in self.range_trees
            )
        )

    def triton_tensor_ndim(self):
        return sum(int(tree.tensor_dim is not None) for tree in self.range_trees)

    def indexing_size_str(self, i):
        sizes = ["None"] * self.triton_tensor_ndim()
        sizes[i] = ":"
        return f"[{', '.join(sizes)}]"

    def dense_size_list(self) -> List[str]:
        sizes = ["1"] * self.triton_tensor_ndim()
        for tree in self.range_trees:
            if tree.tensor_dim is None:
                continue

            if tree.prefix != "r" or self.inside_reduction:
                sizes[tree.tensor_dim] = f"{tree.prefix.upper()}BLOCK"
        return sizes

    def dense_size_str(self):
        sizes = self.dense_size_list()
        return f"[{', '.join(sizes)}]"

    def combine_modular_indexing_pairs(self, index):
        if not isinstance(index, ModularIndexing):
            return index
        x = index.args[0]
        if (tree_node := self.range_tree_nodes.get(x)) is None:
            return index
        new_index = sympy_subs(index, {x: tree_node.expr})
        new_index = V.graph.sizevars.combine_modular_indexing_pairs(new_index)
        # the index now contains xindex/etc, which is nonstandard, fix it up
        return sympy_subs(
            new_index,
            {
                tree_node.root.index_sym(): tree_node.root.lookup(
                    sympy.Integer(1), tree_node.root.numel
                ).symbol()
            },
        )

    def combine_contiguous_dims(self, index: sympy.Expr, tree: IterationRangesRoot):
        if expand_res := V.graph.sizevars.expand_floor_div(index):
            new_index, denominator = expand_res  # type: ignore[misc]
            return FloorDiv(self._combine_contiguous_dims(new_index, tree), denominator)
        else:
            return self._combine_contiguous_dims(index, tree)

    def _combine_contiguous_dims(self, index: sympy.Expr, tree: IterationRangesRoot):
        """
        More aggressive simplification to merge contiguous dims
        """
        if isinstance(index, (sympy.Integer, sympy.Symbol)):
            return index
        index_vars, sizes = tree.vars_and_sizes(index)
        if len(sizes) <= 1:
            return index
        new_sizes, reindex, prune = V.graph.sizevars._simplify_loops(
            index_vars, sizes, index_prevent_reordering([index], index_vars, sizes)
        )
        if new_sizes == sizes:
            return index
        new_index_vars = tree.construct(new_sizes)
        new_index = sympy_subs(index, dict(zip(index_vars, reindex(new_index_vars))))
        return new_index

    def set_last_usage(self, nodes):
        if not self.inside_reduction or self.persistent_reduction:
            return
        self.last_usage = OrderedSet(
            itertools.chain.from_iterable(
                n.last_usage for n in nodes if n is not EnableReduction
            )
        )

    def disable_reduction(self):
        should_flush = self.range_trees[-1].is_loop

        @contextlib.contextmanager
        def ctx():
            if self.numels[-1] == 1:
                assert not self.inside_reduction
                yield
                return
            if should_flush:
                # calling codegen_body() will flush all the pending buffers
                # and write out a reduction loop
                self.codegen_body()
            self.inside_reduction = False
            try:
                yield
                if should_flush:
                    # flush out any code before opening the next loop
                    self.codegen_body()
            finally:
                self.inside_reduction = True

        return ctx()

    def set_ranges(self, *lengths):
        assert len(lengths) == len(self.range_trees)
        return [
            ranges.construct(length)
            for length, ranges in zip(lengths, self.range_trees)
        ]

    @staticmethod
    def _split_iteration_ranges(
        groups: Iterable[sympy.Expr], lengths: Sequence[Sequence[sympy.Expr]]
    ):
        sv = V.graph.sizevars
        new_ranges: List[List[sympy.Expr]] = [[] for _ in groups]
        remaining = [sv.simplify(g) for g in groups]
        var_count = itertools.count()

        def add_range(i, expr):
            expr = sv.simplify(expr)
            if not sv.statically_known_multiple_of(remaining[i], expr):
                raise CantSplit
            # guard on the last item out
            remaining[i] = FloorDiv(remaining[i], expr)
            new_ranges[i].append(expr)
            return next(var_count)

        def make_combined(size, idx1, idx2):
            def getter(flat_vars):
                return size * flat_vars[idx1] + flat_vars[idx2]

            return getter

        return_getters_groups = []
        current_group = 0
        for length_group in lengths:
            return_getters = []
            for size in length_group:
                if sv.statically_known_equals(size, 1):  # type: ignore[arg-type]
                    return_getters.append(lambda _: sympy.Integer(0))
                    continue

                while current_group < len(remaining) and sv.statically_known_equals(
                    remaining[current_group], 1  # type: ignore[arg-type]
                ):
                    # scroll to next group with remaining elements
                    current_group += 1

                if current_group + 1 < len(remaining) and sv.statically_known_gt(
                    size, remaining[current_group]
                ):
                    # need to break size in two
                    if not sv.statically_known_multiple_of(
                        size, remaining[current_group]
                    ):
                        raise CantSplit
                    size1 = remaining[current_group]
                    size2 = FloorDiv(size, remaining[current_group])
                    return_getters.append(
                        make_combined(
                            size2,
                            add_range(current_group, size1),
                            add_range(current_group + 1, size2),
                        )
                    )
                else:
                    return_getters.append(
                        operator.itemgetter(add_range(current_group, size))
                    )
            return_getters_groups.append(return_getters)

        assert all(
            V.graph.sizevars.size_hint(s) == 1 for s in remaining
        ), f"failed to set ranges {remaining} {lengths}"

        return new_ranges, return_getters_groups

    @classmethod
    def is_compatible(
        cls, groups: Iterable[sympy.Expr], lengths: Sequence[Sequence[sympy.Expr]]
    ):
        try:
            cls._split_iteration_ranges(groups, lengths)
            return True
        except CantSplit:
            return False

    def split_and_set_ranges(self, lengths: List[List[sympy.Expr]]):
        """
        We may want to fuse `for i0 in s0*s1` into a tiled kernel with groups (s0, s1).

        To do this we need to split up the iteration space of i0 into something like:
            for i1 in s0:
              for i2 in s1:
                i0 = i1*s1 + i2
                ....

        This function matches and resplits lengths to the groups of
        this kernel to enable tiled + non-tiled fusions.
        """
        groups = [rt.numel for rt in self.range_trees]
        if not self.inside_reduction:
            groups[-1] = sympy.Integer(1)

        if len(lengths) == len(self.range_trees) and all(
            V.graph.sizevars.simplify(sympy_product(x) - g) == 0
            for x, g in zip(lengths, groups)
        ):
            return self.set_ranges(*lengths)

        new_ranges, return_getters_groups = self._split_iteration_ranges(
            groups, lengths
        )
        itervars = list(itertools.chain.from_iterable(self.set_ranges(*new_ranges)))
        return [[fn(itervars) for fn in fns] for fns in return_getters_groups]

    def is_indirect_indexing(self, index: sympy.Expr):
        # tmpX  means indirect indexing
        return free_symbol_is_type(index, SymT.TMP)

    def is_broadcasted(self, index: sympy.Expr):
        # Note. This may not be correct when there is indirect indexing
        if self.is_indirect_indexing(index):
            return False

        index_numels = [1] * len(self.numels)
        for symbol in index.free_symbols:
            if symbol not in self.range_tree_nodes:
                # Non-iterated variables, e.g. strides
                continue
            entry = self.range_tree_nodes[symbol]  # type: ignore[index]
            assert isinstance(entry.parent, IterationRangesRoot)
            index_numels[entry.parent.index] *= entry.length

        # If the index variables only iterate over a subset of the kernel
        # numels, then it must be broadcasted.
        simplify = V.graph.sizevars.simplify
        return any(
            simplify(idx_range) != simplify(iter_range)  # type: ignore[arg-type]
            for idx_range, iter_range in zip(index_numels, self.numels)
        )

    def index_to_str(self, index: sympy.Expr) -> str:
        """
        Convert an index expr to a string that can be used in output code.
        e.g. a sympy expression "s2" may actually appear as "ks1" in the generated kernel.

        Index expressions often need to be passed in as arguments to the triton kernel.
        Rename_indexing and codegen_indexing keep track of the needed indices and add
        new parameters to the function signature.
        """
        if isinstance(index, list):
            return f"[{', '.join(map(self.index_to_str, index))}]"
        return self.kexpr(self.rename_indexing(index))  # type: ignore[call-arg]

    def prepare_indexing(
        self,
        index: sympy.Expr,
    ):
        index = self.simplify_indexing(index)
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        # if simple replacements didn't get rid of floor/ceil, try full subs
        if len(index.atoms(sympy.floor)) or len(index.atoms(sympy.ceiling)):
            index = index.subs(V.graph.sizevars.precomputed_replacements)
        # last resort, if no range vars are in the expr, hoist it
        # TODO instead of trying to blindly find complicated exprs, we should hoist the
        # inputs/outputs sizes and strides, but at the time indexing is generated
        # kernel inputs and outputs are not set yet, we'd need a deeper refactor
        # to do it this way

        if len(index.atoms(sympy.ceiling)):
            for a in index.atoms(sympy.ceiling):
                # for nested exprs, atoms yields top level first (?)
                # so if everything goes fine, lower level replacements will come up empty
                symbols = a.free_symbols
                if len(symbols) > 0 and all(
                    symbol_is_type(s, (SymT.SIZE, SymT.PRECOMPUTED_SIZE))
                    for s in symbols
                ):
                    replacements = {a: V.graph.sizevars.lookup_precomputed_size(a)}
                    index = sympy_subs(index, replacements)

        simp_index = self.simplify_indexing(index)

        # Now that we are done simplifying we can unwrap Identity so that downstream handling
        # for its contained expression will work. previously, tl.full wrapping of sympy.Integer
        # would not occur
        simp_index = (
            simp_index if not isinstance(simp_index, Identity) else simp_index.args[0]
        )

        return self.codegen_indexing(simp_index)

    def active_range_trees(self, reorder=False):
        trees = [
            t for t in self.range_trees if t.prefix != "r" or self.inside_reduction
        ]
        if reorder and len(trees) > 1:
            count = sum(t.prefix in "xyz" for t in trees)
            assert "".join(t.prefix for t in trees[:count]) == "zyx"[-count:], [
                t.prefix for t in trees[:count]
            ]
            trees[:count] = reversed(trees[:count])
        return trees

    def codegen_indexing(self, expr: sympy.Expr):
        expr = V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges())
        for sym in sorted(expr.free_symbols, key=str):
            if sym in self.range_tree_nodes:
                # if indexing expression is complicated, we precompute it on the host side
                # and send the result as a kernel argument
                replacements = {}
                for ps in self.range_tree_nodes[sym].precomputed_args():  # type: ignore[index]
                    replacements[ps] = V.graph.sizevars.lookup_precomputed_size(ps)
                if len(replacements) > 0:
                    self.range_tree_nodes[sym].expr = sympy_subs(  # type: ignore[index]
                        self.range_tree_nodes[sym].expr, replacements  # type: ignore[index]
                    )
                self.range_tree_nodes[sym].codegen()  # type: ignore[index]
        return expr

    def codegen_nan_check(self) -> None:
        raise NotImplementedError("NYI: codegen_nan_check")

    def call_kernel(self, name: str, node: Optional[IRNode] = None) -> None:
        raise NotImplementedError("NYI: call_kernel")

    @contextlib.contextmanager
    def mask_loads(self, mask, value):
        """Context manager to add an additional mask to tl.load/store"""
        prior = self._load_mask
        prior_val = self._load_other
        if prior:
            mask = ops.logical_and(mask, prior)

        mask = OpsWrapper._unwrap(mask)
        self._load_mask = mask
        self._load_other = value
        try:
            # TODO(jansel): do we need a reshape here?
            yield mask
        finally:
            self._load_mask = prior
            self._load_other = prior_val

    def get_strides_of_load(self, index: sympy.Expr):
        """
        This gets the stride of the index for each of the tiling variables
        (technically, it does it at index 0)

        For example, if
        xindex = x0 + 512*x1 + 1024*r0
        x0 = (xindex//512)
        x1 = (xindex % 512)
        r0 = rindex // 1024

        this function would return
        {xindex: 512, rindex: 1024}
        """
        index_to_tile_indexes = {k: v.expr for k, v in self.range_tree_nodes.items()}
        index_in_tile_vars = sympy_subs(index, index_to_tile_indexes)  # type: ignore[arg-type]
        strides = {}
        for range_tree in self.range_trees:
            s = sympy_index_symbol(range_tree.name)
            strides[s] = sympy_subs(index_in_tile_vars, {s: 1}) - sympy_subs(
                index_in_tile_vars, {s: 0}
            )
        return strides

    @staticmethod
    def _map_tuple_or_scalar(fn, value):
        if isinstance(value, tuple):
            return tuple(map(fn, value))
        return fn(value)

    def estimate_kernel_num_bytes(self):
        """
        Try the best to estimate the total size (in bytes) of the
        kernel's inputs and outputs, which is used for estimating the memory
        throughput of this kernel. This information is used for checking how
        far we are from the peak memory bandwidth. It's important that
        we want to avoid overestimating the sizes of the inputs and outputs,
        because it can wrongfully give us a very large memory traffic value,
        which may be even larger than the theoretical bandwidth and thus
        become very misleading. This is particularly problematic for cases
        where we slice some inputs. In those cases, we should only count
        the size of the "slices" instead of the original inputs, because
        only the slices contribute to the real memory traffic.
        """
        nbytes = []
        ninplace_args = len(unique(self.args.inplace_buffers.values()))
        _, call_args, _, _ = self.args.python_argdefs()

        # For pointwise and reduction kernels, this is the upper-bound numels
        # for the output buffer.
        # FIXME: This is not exactly right for cases like below:
        #    def foo(tensor0, tensor1):
        #        x0 = narrow(tensor0)
        #        return cat(x0, tensor1)
        # For this example, we will end up overestimate the size for the
        # slice s0. Potentially, we could have precise inputs information
        # if we maintained the original inputs of the Pointwise kernel created
        # for the "cat". However, I think it might be a bit overwhelming that
        # we add such complexity only for handling some particular cases for
        # benchmarking.
        out_numel = V.graph.sizevars.size_hint(sympy_product(self.numels))
        for i, arg in enumerate(call_args):
            # "buf" may be narrowed. In this case, the number of memory accesses
            # should be estimated based on the reinterpreted layout.
            # On the other hand, buf may be broadcasted. In this case,
            # counting the size of the underline storage would give us
            # a better estimation in terms of memory accesses.
            if arg not in self.buf_accesses:
                nbytes.append(0)
                continue
            arg_numel = V.graph.get_numel(arg)
            buf_size = V.graph.sizevars.size_hint(arg_numel)
            if buf_size > out_numel:
                # This arg points to a buf that has been sliced.
                # We need to count each individual slice to have
                # a better estimation.
                indices: OrderedSet[Any] = OrderedSet()
                no_index_dep_count = 0
                for dep in self.buf_accesses[arg]:
                    if isinstance(dep, (StarDep, WeakDep)):
                        indices.add(f"no_index_dep_{no_index_dep_count}")
                        no_index_dep_count += 1
                    else:
                        indices.add(dep.index)
                numel = len(indices) * out_numel
            else:
                numel = buf_size
            dtype = V.graph.get_dtype(arg)
            dtype_size = get_dtype_size(dtype)
            nbytes.append(numel * dtype_size * (1 + int(i < ninplace_args)))
        return sum(nbytes)

    def warn_mix_layout(self, kernel_name):
        """
        Print message if the kernel have mixed layout inputs.
        Only care about 4D tensor for now.
        """
        if (
            len(self.args.input_buffers) == 1
            and len(self.args.output_buffers) == 1
            and len(self.args.inplace_buffers) == 0
        ):
            # even if input buffer and output buffer have different layout,
            # this can be a layout conversion kernel. No need to warn for
            # the mix layouts.
            return

        argdefs, call_args, signature, _ = self.args.python_argdefs()
        uniform_stride_order = None
        for arg_name in call_args:
            buf = V.graph.try_get_buffer(arg_name)
            if buf and len(buf.layout.size) == 4:
                # ignore the tensor if only 1 dimension is non-zero
                if len([x for x in buf.layout.size if x == 1]) == 3:
                    continue
                stride_order = ir.get_stride_order(buf.layout.stride)
                if uniform_stride_order is None:
                    uniform_stride_order = stride_order
                elif uniform_stride_order != stride_order:
                    msg = yellow_text(
                        f"Expected stride order {uniform_stride_order}, but found stride order"
                        + f" {stride_order} for kernel {kernel_name}"
                    )
                    log.warning(msg)

                    stride_order_list = [
                        ir.get_stride_order(V.graph.get_buffer(name).layout.stride)
                        if V.graph.try_get_buffer(name)
                        else None
                        for name in call_args
                    ]
                    size_list = [
                        V.graph.get_buffer(name).layout.size
                        if V.graph.try_get_buffer(name)
                        else None
                        for name in call_args
                    ]
                    source_list = [
                        "GraphInput"
                        if name in V.graph.graph_inputs
                        else "IntermediateBuffer"
                        if name in V.graph.name_to_buffer
                        else None
                        for name in call_args
                    ]

                    msg = yellow_text(
                        f"  param names {argdefs}\n  buf names {call_args}\n  strides {stride_order_list}"
                        + f"\n  sizes {size_list}\n  sources {source_list}\n"
                    )
                    log.warning(msg)
                    return
        msg = green_text(
            f"All the inputs for the triton kernel {kernel_name} have uniform layout"
        )
        log.warning(msg)

    def welford_reduce_fallback(self, dtype, value):
        sum_ = ops.reduction(dtype, dtype, "sum", value)
        self.inside_reduction = False
        rnumel = ops.index_expr(self.numels[-1], dtype)
        mean = ops.truediv(sum_, rnumel)

        self.inside_reduction = True
        dx = ops.sub(value, mean)
        dx2 = ops.mul(dx, dx)
        m2 = ops.reduction(dtype, dtype, "sum", dx2)
        return OpsWrapper._unwrap((mean, m2, rnumel))

    def codegen_kernel(self):
        raise NotImplementedError

    def codegen_body(self):
        pass

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry):
        pass


class SIMDScheduling(BaseScheduling):
    kernel_type = SIMDKernel  # override in subclass
    int32_type = "torch.int32"
    int64_type = "torch.int64"

    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    def group_fn(self, sizes):
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    def can_fuse(self, node1, node2):
        """
        Hook called by Scheduler to determine if the Triton backend
        can fuse node1 and node2.  These nodes might already be
        FusedSchedulerNodes.
        """
        if isinstance(node1, scheduler.ForeachKernelSchedulerNode) or isinstance(
            node2, scheduler.ForeachKernelSchedulerNode
        ):
            return scheduler.ForeachKernelSchedulerNode.can_fuse(node1, node2)

        _, (numel1, rnumel1) = node1.group
        _, (numel2, rnumel2) = node2.group
        why = WhyNoFuse(node1, node2)

        if node1.is_split_scan() and not node2.is_split_scan():
            if node2.is_reduction():
                why("Split scan cannot fuse with reductions")
        elif node2.is_split_scan() and not node1.is_split_scan():
            if node1.is_reduction():
                why("Split scan cannot fuse with reductions")

        if node1.is_reduction() and node2.is_reduction():
            reduction_can_fuse = numel1 == numel2 and rnumel1 == rnumel2
            if not reduction_can_fuse:
                why(
                    "numel/rnumel mismatch (reduce) (%s, %s), (%s, %s)",
                    numel1,
                    numel2,
                    rnumel1,
                    rnumel2,
                )
            return reduction_can_fuse

        if not node1.is_reduction() and not node2.is_reduction():
            if not (numel1 == numel2 and rnumel1 == rnumel2):
                why(
                    "numel/rnumel mismatch (non-reduce) (%s, %s), (%s, %s)",
                    numel1,
                    numel2,
                    rnumel1,
                    rnumel2,
                )
                return False

            if node1.is_template():
                # Only allow fusion for TritonTemplates for now.
                # Fusion for CUDATemplates are not supported.
                is_triton_template = isinstance(node1.node, TritonTemplateBuffer)
                if not is_triton_template:
                    why("node1 is not TritonTemplateBuffer")
                return is_triton_template

            # check for a bad combined tiling
            tiling1 = self.select_tiling(node1.get_nodes(), numel1, rnumel1)
            tiling2 = self.select_tiling(node2.get_nodes(), numel1, rnumel1)
            tiling3 = self.select_tiling(
                node1.get_nodes() + node2.get_nodes(), numel1, rnumel1
            )
            if config.triton.tiling_prevents_pointwise_fusion:
                cond = True
                if len(tiling1) > 2:
                    if len(tiling2) > 2:
                        cond = tiling1 == tiling2 == tiling3
                    else:
                        cond = tiling1 == tiling3
                elif len(tiling2) > 2:
                    cond = tiling2 == tiling3
                if not cond:
                    why(
                        "tiling mismatch (%s, %s, %s)",
                        tiling1,
                        tiling2,
                        tiling3,
                    )
                    return False

            return True

        if not node1.is_reduction() and node2.is_reduction():
            assert rnumel1 == 1 and rnumel2 != 1
            if numel1 == numel2 * rnumel2:
                if not all(
                    SIMDKernel.is_compatible((numel2, rnumel2), n.get_ranges())
                    for n in node1.get_nodes()
                ):
                    why("nodes numel/rnumel incompatibility")
                    return False
                if (
                    config.triton.tiling_prevents_reduction_fusion
                    and not node1.is_template()
                ):
                    is_reduction_tiling_valid = self.select_tiling(
                        node1.get_nodes(), numel1
                    ) in (
                        (numel1, 1),
                        (numel2, rnumel2, 1),
                    )
                    if not is_reduction_tiling_valid:
                        why("invalid tiling for reduction")
                    return is_reduction_tiling_valid
                return True

            if numel1 != numel2:
                why("nodes numel incompatibility")
            return numel1 == numel2

        assert node1.is_reduction() and not node2.is_reduction()
        # swap args to hit the case above
        return self.can_fuse_horizontal(node2, node1)

    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse

    def generate_node_schedule(self, nodes, numel, rnumel):
        node_schedule: List[Any] = []
        done: OrderedSet[scheduler.BaseSchedulerNode] = OrderedSet()
        # Writes with a reduced shape, meaning they are only present once the
        # reduction loop has ended
        not_ready_yet_nodes: OrderedSet[str] = OrderedSet()

        def fits_in_main_body(n):
            _, (node_numel, node_rnumel) = n.group
            return (node_numel == numel and node_rnumel == rnumel) or (
                node_numel == numel * rnumel and node_rnumel == 1
            )

        def fits_outside_reduction(n):
            _, (node_numel, node_rnumel) = n.group
            return node_numel == numel and node_rnumel == 1 and rnumel != 1

        def schedule_node_in_loop(n):
            done.add(n)
            node_schedule.append(n)
            # A scan is modelled as a reduction in the scheduler but has a
            # full sized output that can be used inside the loop body
            if (
                n.is_reduction()
                and isinstance(n, scheduler.SchedulerNode)
                and isinstance(n.node, ir.ComputedBuffer)
                and not isinstance(n.node.data, ir.Scan)
            ):
                not_ready_yet_nodes.add(n.get_name())

        @contextlib.contextmanager
        def end_current_reduction_loop():
            if node_schedule and node_schedule[-1] is EnableReduction:
                node_schedule.pop()
            else:
                node_schedule.append(DisableReduction)
            yield
            node_schedule.append(EnableReduction)
            not_ready_yet_nodes.clear()

        def requires_closing_previous_reduction(node, node_schedule):
            if rnumel == 1:
                return False
            if not not_ready_yet_nodes & node.ancestors:
                return False
            assert node_schedule and not isinstance(
                node_schedule[-1], (EnableReduction, DisableReduction)
            )
            return bool(not_ready_yet_nodes)

        for index, node in enumerate(nodes):
            if node in done:
                continue
            done.add(node)

            if fits_in_main_body(node):
                if requires_closing_previous_reduction(node, node_schedule):
                    with end_current_reduction_loop():
                        pass  # need to start a new reduction loop

                schedule_node_in_loop(node)
            elif fits_outside_reduction(node):
                with end_current_reduction_loop():
                    node_schedule.append(node)
            else:
                raise NotImplementedError(
                    f"unexpected group: ({numel}, {rnumel}) != {node.group[1]}"
                )

        return node_schedule

    def codegen_node(
        self, node: Union[scheduler.FusedSchedulerNode, scheduler.SchedulerNode]
    ):
        """
        Given a set of pre-fused nodes, generate a Triton kernel.
        """

        nodes: List[scheduler.SchedulerNode] = node.get_nodes()  # type: ignore[assignment]

        _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group

        node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
        buf_accesses = collections.defaultdict(list)
        for node in nodes:
            for access in node.read_writes.reads | node.read_writes.writes:
                buf_accesses[access.name].append(access)

        schedule_log.debug("Schedule:\n %s", node_schedule)

        return self.codegen_node_schedule(node_schedule, buf_accesses, numel, rnumel)

    @staticmethod
    def reduction_hint(node):
        assert node.is_reduction()
        if all(
            dep.is_contiguous()
            for dep in itertools.chain(node.read_writes.reads, node.read_writes.writes)
        ):
            return ReductionHint.INNER
        else:
            return node.node.data.reduction_hint

    @staticmethod
    def can_use_32bit_indexing(
        numel: sympy.Expr, buffers: Iterable[Union[ir.Buffer, ir.TensorBox]]
    ) -> bool:
        int_max = torch.iinfo(torch.int32).max
        size_hint = V.graph.sizevars.size_hint
        has_hint = V.graph.sizevars.shape_env.has_hint

        def within_32bit(e):
            # Allow for unhinted e as long as we can still statically prove
            # (e.g., via ValueRanges) that it is still in bounds
            if V.graph.sizevars.is_expr_static_and_true(e <= int_max):
                return True
            # Otherwise, the hint MUST exist and be in range
            return has_hint(e) and size_hint(e) <= int_max

        if not within_32bit(numel):
            return False

        # Any use of a MultiOutputLayout will create a buffer with a
        # Layout whose sizes are accounted for
        buf_sizes = [
            buf.get_layout().storage_size()
            for buf in buffers
            if not isinstance(buf.get_layout(), ir.MultiOutputLayout)
        ]

        if not all(within_32bit(size) for size in buf_sizes):
            return False

        # Only install guards for 32-bit indexing as there is no correctness
        # issue with using 64-bit for everything
        V.graph.sizevars.guard_leq(numel, int_max)  # type: ignore[arg-type]
        for size in buf_sizes:
            V.graph.sizevars.guard_leq(size, int_max)  # type: ignore[arg-type]
        return True

    @classmethod
    def select_index_dtype(cls, node_schedule, numel, reduction_numel):
        # Gather all used buffer names
        buffer_names: OrderedSet[str] = OrderedSet()
        for node in node_schedule:
            if not isinstance(node, scheduler.BaseSchedulerNode):
                continue

            buffer_names.update(node.get_buffer_names())
            buffer_names.update(node.used_buffer_names())

        # Get buffers objects

        def _get_buffer(name: str) -> Union[ir.Buffer, ir.TensorBox]:
            buf = V.graph.get_buffer(name)
            if buf is None:
                raise RuntimeError(f"Failed to find buffer matching name {name}")
            return buf

        buffers = [V.graph.get_buffer(name) for name in buffer_names]

        # In theory we can separately check xnumel and rnumel are <= int_max
        # but some indexers do use the full linear index so we need to be
        # conservative here.
        total_numel = numel * reduction_numel

        if SIMDScheduling.can_use_32bit_indexing(total_numel, buffers):
            return cls.int32_type
        return cls.int64_type

    def has_non_contiguous_pw_in_reduction_kernel(self, node_schedule, numel, rnumel):
        pointwise_nodes = list(
            filter(
                lambda n: n not in (EnableReduction, DisableReduction)
                and not n.is_reduction()
                and n.group[1][0] == numel * rnumel,
                node_schedule,
            )
        )
        for node in pointwise_nodes:
            # An index can be an integer when loading a random seed.
            if not all(
                not isinstance(dep, MemoryDep)
                or dep.is_contiguous()
                or isinstance(dep.index, (sympy.Integer, int))
                or dep.stride1_for_last_dim()
                for dep in itertools.chain(
                    node.read_writes.reads, node.read_writes.writes
                )
            ):
                return True
        return False

    def get_kernel_args(self, node_schedule, numel, reduction_numel):
        reductions = list(
            filter(
                lambda n: n not in (EnableReduction, DisableReduction)
                and n.is_reduction(),
                node_schedule,
            )
        )
        if len(reductions) > 0:
            hints = [self.reduction_hint(n) for n in reductions]
            if hints.count(hints[0]) == len(hints):
                reduction_hint_val = hints[0]
            else:
                reduction_hint_val = ReductionHint.DEFAULT

            if (
                reduction_hint_val == ReductionHint.INNER
                and self.has_non_contiguous_pw_in_reduction_kernel(
                    node_schedule, numel, reduction_numel
                )
            ):
                reduction_hint_val = ReductionHint.DEFAULT
        else:
            reduction_hint_val = ReductionHint.DEFAULT

        mutations: OrderedSet[str] = OrderedSet()
        for node in node_schedule:
            if node in (DisableReduction, EnableReduction):
                continue

            for buf in node.get_outputs():
                mutations.update(buf.get_mutations())

        index_dtype = self.select_index_dtype(node_schedule, numel, reduction_numel)

        return reduction_hint_val, mutations, index_dtype

    def codegen_node_schedule(
        self, node_schedule, buf_accesses, numel, reduction_numel
    ):
        from torch._inductor.codegen.triton_split_scan import TritonSplitScanKernel

        tiled_groups = self.select_tiling(node_schedule, numel, reduction_numel)
        (
            reduction_hint_val,
            mutations,
            index_dtype,
        ) = self.get_kernel_args(node_schedule, numel, reduction_numel)

        is_split_scan = any(
            isinstance(node, BaseSchedulerNode) and node.is_split_scan()
            for node in node_schedule
        )
        kernel_type: type = self.kernel_type
        if is_split_scan and issubclass(TritonSplitScanKernel, kernel_type):
            kernel_type = TritonSplitScanKernel

        kernel_args = tiled_groups
        kernel_kwargs = dict(
            reduction_hint=reduction_hint_val,
            mutations=mutations,
            index_dtype=index_dtype,
        )

        def _node_has_sort(node):
            if node in (EnableReduction, DisableReduction):
                return False

            sort_nodes = node._body.root_block.graph.find_nodes(
                op="call_method", target="sort"
            )
            return bool(sort_nodes)

        # ops.sort only works with persistent reduction, and is not bandwidth bound anyway
        # so taking the hit of non-coalesced loads is okay
        has_sort = any(_node_has_sort(node) for node in node_schedule)
        if has_sort:
            kernel_kwargs["override_persistent_reduction"] = True

        kernel = kernel_type(
            *kernel_args,
            **kernel_kwargs,
        )
        kernel.buf_accesses = buf_accesses

        kernel2: Optional[SIMDKernel] = None
        if kernel.persistent_reduction and config.triton.multi_kernel and not has_sort:
            kernel2 = self.kernel_type(
                *kernel_args,
                **kernel_kwargs,
                override_persistent_reduction=False,
            )
            self.codegen_node_schedule_with_kernel(node_schedule, kernel2)
            with V.set_kernel_handler(kernel2):
                src_code2 = kernel2.codegen_kernel()
            kernel_name2 = self.define_kernel(src_code2, node_schedule, kernel)
            kernel2.kernel_name = kernel_name2
            kernel2.code_hash = code_hash(src_code2)

            # Keep buffers needed by the non-persistent reduction so both
            # kernels have the same arguments
            kernel.must_keep_buffers = set(kernel2.must_keep_buffers)

        self.codegen_node_schedule_with_kernel(node_schedule, kernel)

        with V.set_kernel_handler(kernel):
            src_code = kernel.codegen_kernel()

        kernel_name = self.define_kernel(src_code, node_schedule, kernel)
        log.debug("Generating kernel code with kernel_name: %s", kernel_name)
        kernel.kernel_name = kernel_name
        kernel.code_hash = code_hash(src_code)

        final_kernel = MultiKernel([kernel, kernel2]) if kernel2 is not None else kernel

        with V.set_kernel_handler(final_kernel):
            for node in node_schedule:
                if node not in (EnableReduction, DisableReduction):
                    node.mark_run()

        self.codegen_comment(node_schedule)

        _, call_args, arg_signatures, _ = (
            final_kernel.args.python_argdefs()
            if not isinstance(final_kernel, MultiKernel)
            else [None, [], None, None]
        )
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(
            call_args, kernel_name, arg_signatures, final_kernel
        )
        with debug_printer_manager:
            final_kernel.call_kernel(final_kernel.kernel_name)

        if config.nan_asserts:
            final_kernel.codegen_nan_check()
        if config.warn_mix_layout:
            final_kernel.warn_mix_layout(kernel_name)

        V.graph.removed_buffers |= final_kernel.removed_buffers
        V.graph.inplaced_to_remove |= final_kernel.inplaced_to_remove

        if (
            V.graph.wrapper_code.supports_intermediate_hooks
            and config.generate_intermediate_hooks
        ):
            # Not every node in the schedule will actually be live on output;
            # we can't check dead buffers.
            live_outs = kernel.args.live_output_buffers()
            for node in node_schedule:
                if not isinstance(node, scheduler.BaseSchedulerNode):
                    continue
                name = node.get_name()
                if name not in live_outs:
                    continue
                assert node.node is not None
                origin_node = node.node.get_origin_node()
                if origin_node is not None:
                    counters["inductor"]["intermediate_hooks"] += 1
                    V.graph.wrapper_code.writeline(
                        f"run_intermediate_hooks({origin_node.name!r}, {name})"
                    )

        self.scheduler.free_buffers()

    def codegen_node_schedule_with_kernel(self, node_schedule, kernel):
        def current_reduction_nodes(nodes):
            return itertools.takewhile(lambda n: n is not DisableReduction, nodes)

        with kernel:
            stack = contextlib.ExitStack()
            kernel.set_last_usage(current_reduction_nodes(node_schedule))
            all_indexing = {}

            # First pass to collect indexing and decide inplace updates
            for node in node_schedule:
                if node is DisableReduction:
                    stack.enter_context(kernel.disable_reduction())
                elif node is EnableReduction:
                    stack.close()
                else:
                    node.decide_inplace_update()
                    index_vars = kernel.split_and_set_ranges(node.get_ranges())
                    all_indexing.update(
                        dict.fromkeys(
                            node._body.indexing_from_args(index_vars).values()
                        )
                    )

            kernel.finalize_indexing(all_indexing.keys())

            # Second pass to do codegen
            for i, node in enumerate(node_schedule):
                if node is DisableReduction:
                    stack.enter_context(kernel.disable_reduction())
                elif node is EnableReduction:
                    stack.close()
                    kernel.set_last_usage(current_reduction_nodes(node_schedule[i:]))
                else:
                    # TODO - use split ranges ?
                    indexing_dtype_strength_reduction(node._body)
                    index_vars = kernel.split_and_set_ranges(node.get_ranges())
                    node.codegen(index_vars)

    def codegen_template(
        self, template_node, epilogue_nodes, only_gen_src_code=False
    ) -> Optional[str]:
        """
        Codegen a triton template

        If `only_gen_src_code` the src code will be returned instead of codegen'd into the wrapper
        """
        _, (numel, rnumel) = template_node.group
        assert rnumel == 1
        kernel, render = template_node.node.make_kernel_render(template_node.node)
        with kernel:
            if not only_gen_src_code:
                for node in [template_node, *epilogue_nodes]:
                    node.mark_run()
            partial_code = render()
            with kernel.set_subgraph_body("<STORE_OUTPUT>"):
                for node in epilogue_nodes:
                    node.codegen(kernel.split_and_set_ranges(node.get_ranges()))

        if not isinstance(partial_code, str):
            partial_code.finalize_hook("<DEF_KERNEL>")
            partial_code.finalize_hook("<ARGDEFS>", strict=False)
        # finalize must be called after adding epilogue above
        with V.set_kernel_handler(kernel):
            # TODO: Maybe unify CUDATemplateKernel to also use PartialRender for flexible epilogue fusion.
            with kernel.set_subgraph_body("<STORE_OUTPUT>"):
                if isinstance(partial_code, str):
                    src_code = partial_code
                else:
                    partial_code.finalize_hook("<STORE_OUTPUT>")
                    src_code = partial_code.code
            node_schedule = [template_node, *epilogue_nodes]

            if config.benchmark_kernel:
                num_gb = kernel.estimate_kernel_num_bytes() / 1e9
                grid_args = V.graph.sizevars.size_hints(kernel.call_sizes)
                assert kernel.meta is not None, "meta is None"
                grid = kernel.grid_fn(*grid_args, kernel.meta)
                src_code = (
                    f"{kernel.imports_for_benchmark_kernel()}\n"
                    f"{src_code}\n"
                    f"{kernel.codegen_kernel_benchmark(num_gb, grid).getvalue()}"
                )

            if only_gen_src_code:
                return src_code

            kernel_name = self.define_kernel(src_code, node_schedule, kernel)

        self.codegen_comment(node_schedule)

        # debug printing values of intermediate tensors
        _, call_args, arg_signatures, _ = kernel.args.python_argdefs()
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(
            call_args, kernel_name, arg_signatures, kernel
        )
        with debug_printer_manager:
            kernel.call_kernel(kernel_name, template_node.node)

        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
        self.scheduler.free_buffers()
        return None

    def codegen_sync(self):
        V.graph.wrapper_code.writeline(V.graph.device_ops.synchronize())

    def generate_combo_kernel_code(
        self,
        subkernel_nodes: List[BaseSchedulerNode],
        custom_part_algorithm: bool,
        enable_autotune: bool,
        mixed_sizes: bool,
        only_gen_src_code: bool = False,
    ) -> List[Tuple[str, Any, Any]]:
        from .triton_combo_kernel import ComboKernel

        fused_node_lists = [node.get_nodes() for node in subkernel_nodes]
        subkernel_map, node_schedule_map = {}, {}
        for pn, nodes in zip(subkernel_nodes, fused_node_lists):
            _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group
            node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
            tiled_groups = self.select_tiling(node_schedule, numel, rnumel)
            node_schedule_map[pn] = node_schedule, tiled_groups, numel, rnumel
            (
                reduction_hint_val,
                mutations,
                index_dtype,
            ) = self.get_kernel_args(node_schedule, numel, rnumel)
            subkernel_map[pn] = ComboKernel.create_triton_kernel(
                *tiled_groups,
                reduction_hint=reduction_hint_val,
                mutations=mutations,
                index_dtype=index_dtype,
                optimize_mask=not mixed_sizes,
            )

        partitions = ComboKernel.horizontal_partition(
            nodes=subkernel_nodes,
            triton_scheduling=self,
            custom_algorithm=custom_part_algorithm,
            kernel_map=subkernel_map,
            node_info_map=node_schedule_map,
        )
        log.debug(
            "ComboKernels: %d nodes partitioned into %s groups",
            len(subkernel_nodes),
            [len(p) for p in partitions],
        )
        kernel_code_list = []
        for node_group in partitions:
            fused_node_lists = [node.get_nodes() for node in node_group]
            kernel = ComboKernel(
                enable_autotune=enable_autotune,
                mixed_sizes=mixed_sizes,
            )

            for pn, nodes in zip(node_group, fused_node_lists):
                if only_gen_src_code:
                    # empty last_usage. May cause more aggressive 'evict_last'. Should be fine.
                    for n in nodes:
                        n.last_usage = OrderedSet()
                self.codegen_node_schedule_with_kernel(
                    node_schedule_map[pn][0],
                    kernel.create_sub_kernel(subkernel_map[pn]),
                )
                subkernel = subkernel_map[pn]
                node_schedule = node_schedule_map[pn][0]
                if not only_gen_src_code:
                    with V.set_kernel_handler(subkernel):  # type: ignore[call-arg]
                        for node in node_schedule:
                            if node not in (EnableReduction, DisableReduction):
                                node.mark_run()
                V.graph.removed_buffers |= subkernel.removed_buffers
                V.graph.inplaced_to_remove |= subkernel.inplaced_to_remove

            src_code = kernel.codegen_kernel()
            kernel_code_list.append((src_code, kernel, node_group))
        return kernel_code_list

    def codegen_combo_kernel(self, combo_kernel_node):
        subkernel_nodes = combo_kernel_node.get_subkernel_nodes()
        custom_part_algorithm = combo_kernel_node.use_custom_partition_algo
        enable_autotune = combo_kernel_node.enable_autotune
        mixed_sizes = config.combo_kernel_allow_mixed_sizes > 1 or (
            config.combo_kernel_allow_mixed_sizes == 1 and custom_part_algorithm
        )

        kernel_code_list = self.generate_combo_kernel_code(
            subkernel_nodes, custom_part_algorithm, enable_autotune, mixed_sizes
        )

        for src_code, kernel, _ in kernel_code_list:
            kernel_name = self.define_kernel(src_code, [combo_kernel_node], kernel)
            self.codegen_comment([combo_kernel_node])
            log.debug("ComboKernels: generated kernel %s.", kernel_name)
            kernel.call_kernel(V.graph.wrapper_code, kernel_name)

        self.scheduler.free_buffers()

    @staticmethod
    @functools.lru_cache(32)
    def candidate_tilings(node):
        ranges, reduction_ranges = node.get_ranges()
        if len(ranges) <= 1:
            return ()

        rw = node.pointwise_read_writes()
        assert len(rw.range_vars) == len(ranges)

        # isinstance(dep, MemoryDep): this filters out StarDeps. StarDeps refer to reads
        # that need to access the entire tensor; they don't contribute read indexing
        # information (and practically, they don't have dep.index so they can't be used
        # for stride_hints below
        dep_sources = [rw.reads, rw.writes]
        assert all(
            isinstance(dep, (MemoryDep, StarDep))
            for dep in itertools.chain.from_iterable(dep_sources)
        )
        deps = [
            dep
            for dep in itertools.chain.from_iterable(dep_sources)
            if dep.name not in V.graph.removed_buffers and isinstance(dep, MemoryDep)
        ]
        write_names = {dep.name for dep in rw.writes}

        tilings: List[CandidateTiling] = []

        for dep in deps:
            strides = V.graph.sizevars.stride_hints(dep.index, rw.range_vars)
            assert len(strides) == len(ranges)
            try:
                split = strides.index(1) + 1
                if split == len(ranges):
                    continue
                if all(s == 0 for s in strides[split:]):
                    # if this is a broadcasted tensor and all dimensions after split are broadcast,
                    # this is not a real split
                    continue

            except ValueError:
                continue
            tiled_groups = (
                V.graph.sizevars.simplify(sympy_product(ranges[:split])),
                V.graph.sizevars.simplify(sympy_product(ranges[split:])),
            )
            # score by number of elements
            score = V.graph.sizevars.size_hint(
                sympy_product(
                    size for size, stride in zip(ranges, strides) if stride != 0
                )
            )
            if dep.name in write_names:
                # ngimel said contiguous writes is more important than reads
                score *= 2
            if CandidateTiling.is_good_size(tiled_groups[0]):
                score *= 2
            if CandidateTiling.is_good_size(tiled_groups[1]):
                score *= 2

            if (
                V.graph.sizevars.size_hint(
                    score - sympy_product(itertools.chain(ranges, reduction_ranges))
                )
                >= 0
            ):
                tilings.append(CandidateTiling(tiled_groups, score, dep.name))
        return tilings

    @classmethod
    def select_tiling(cls, node_schedule, numel, reduction_numel=sympy.Integer(1)):
        """
        Heuristics to decide how to tile kernels.
        Currently, we tile based on stride-1 dimensions.

        Returns:
            `(tile1, tile2, reduction_numel)` s.t. `tile1 * tile2 == numel`

        """
        if reduction_numel != 1 or config.triton.max_tiles <= 1:
            # TODO(jansel): should we tile reductions?
            # do perf hint here if stride-1 dim is not being reduced
            if perf_hint_log.level <= logging.WARNING:
                for node in EnableReduction.filter(node_schedule):
                    if len(cls.candidate_tilings(node)) > 0:
                        perf_hint_log.info("reduction over non-contiguous dims")
                        break
            return (numel, reduction_numel)

        seen_names: OrderedSet[str] = OrderedSet()
        candidate_tiles: Counter[Any] = collections.Counter()
        for node in EnableReduction.filter(node_schedule):
            for tiling in cls.candidate_tilings(node):
                if tiling.name in seen_names:
                    continue
                seen_names.add(tiling.name)
                candidate_tiles[tiling.tiling] += tiling.score

        ranked_tilings = [tiling for tiling, score in candidate_tiles.most_common()]

        if config.triton.max_tiles >= 3:
            # Consider adding a third dimension of tiling, but only
            # when a1 is a multiple of b1; otherwise, you have a lot
            # of stragglers which is annoying to generate code for.
            #
            # NB: More than three max tiles is not enabled by default.

            # Add one 3D tiling choice
            for i in range(1, len(ranked_tilings)):
                a0, a1 = ranked_tilings[0]
                b0, b1 = ranked_tilings[i]
                if V.graph.sizevars.size_hint(a1 - b1) == 0:
                    continue
                if V.graph.sizevars.size_hint(a1 - b1) < 0:
                    # swap so a0 is bigger
                    a0, a1 = ranked_tilings[i]
                    b0, b1 = ranked_tilings[0]
                assert V.graph.sizevars.size_hint(a1 - b1) > 0
                if V.graph.sizevars.statically_known_multiple_of(a1, b1):
                    tiling = (a0, FloorDiv(a1, b1), b1)
                    ranked_tilings = [tiling] + ranked_tilings
                    break  # only 1 choice for now

        if len(ranked_tilings) > 1:
            perf_hint_log.info("possibly bad tiling: %s", ranked_tilings)

        # Optionally, prefer tiling into as many dimensions as possible.
        if config.triton.prefer_nd_tiling:
            # Get candidate tilings from the node ranges.
            node_ranges = [
                node.get_ranges()[0]
                for node in EnableReduction.filter(node_schedule)
                if isinstance(node, scheduler.SchedulerNode)
            ]
            new_tilings: OrderedSet[Tuple[sympy.Expr]] = OrderedSet()
            for node_range in node_ranges:
                # Collapse leading dims, to fit in the maximum dimensionality.
                num_leading_dims = max(0, len(node_range) - config.triton.max_tiles)
                first_trailing_dim = num_leading_dims + 1
                collapsed_leading_dim = sympy_product(node_range[:first_trailing_dim])
                tiling = [collapsed_leading_dim] + list(node_range[first_trailing_dim:])
                new_tilings.add(tuple(tiling))

            # Rank tilings by the number of dimensions. E.g., prefer 2D to 1D.
            # Since this is a stable sort, ties are broken by schedule order.
            ranked_new_tilings = sorted(new_tilings, key=len, reverse=True)
            ranked_tilings = ranked_new_tilings + ranked_tilings

        for tiled_groups in ranked_tilings:
            new_groups = (*tiled_groups, reduction_numel)
            if all(
                SIMDKernel.is_compatible(new_groups, node.get_ranges())
                for node in node_schedule
                if isinstance(node, scheduler.SchedulerNode)
            ):
                return new_groups

        return (numel, reduction_numel)

    def flush(self):
        pass

    def ready_to_flush(self) -> bool:
        return False

    def generate_kernel_code_from_nodes(self, nodes, benchmark_kernel=False):
        @dataclasses.dataclass
        class LastUsageHolder:
            n: Any
            last_usage: Any

            def __del__(self) -> None:
                self.n.last_usage = self.last_usage

        last_usage_holders = [LastUsageHolder(n, n.last_usage) for n in nodes]

        # empty last_usage. May cause more aggressive 'evict_last'. Should be fine.
        for n in nodes:
            n.last_usage = OrderedSet()

        if not nodes[0].is_template():
            _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group
            node_schedule = self.generate_node_schedule(nodes, numel, rnumel)

            tiled_groups = self.select_tiling(node_schedule, numel, rnumel)
            reduction_hint_val, mutations, index_dtype = self.get_kernel_args(
                node_schedule, numel, rnumel
            )

            kernel = self.kernel_type(
                *tiled_groups,
                reduction_hint=reduction_hint_val,
                mutations=mutations,
                index_dtype=index_dtype,
            )

            self.codegen_node_schedule_with_kernel(node_schedule, kernel)
            with config.patch(
                "benchmark_kernel", benchmark_kernel
            ), V.set_kernel_handler(kernel):
                src_code = kernel.codegen_kernel()
        else:
            template_node = nodes[0]
            epilogue_nodes = nodes[1:]

            with config.patch("benchmark_kernel", benchmark_kernel):
                src_code = self.codegen_template(
                    template_node, epilogue_nodes, only_gen_src_code=True
                )

        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), "triton_")
        return src_code

    def codegen_comment(self, node_schedule):
        pass

    def define_kernel(self, src_code, node_schedule, kernel):
        raise NotImplementedError


@dataclasses.dataclass
class CandidateTiling:
    tiling: Tuple[sympy.Expr, sympy.Expr]
    score: int  # higher is better
    name: Optional[str] = None

    @staticmethod
    def is_good_size(s):
        """Somewhat arbitrary heuristic used to boost scores for some sizes"""
        s = V.graph.sizevars.size_hint(s)
        return s >= 32 and (s % 32 == 0)


class DisableReduction:
    """
    Marker to invoke `kernel.disable_reduction()`.  This closes a
    reduction loop and allows for pointwise ops to occur on the output
    of a reduction.
    """


class EnableReduction:
    """
    Marker to end a DisableReduction block.
    """

    @staticmethod
    def filter(node_schedule):
        """
        Get the nodes from node_schedule skipping those in a
        DisableReduction block.
        """
        disabled = False
        for node in node_schedule:
            if node in (EnableReduction, DisableReduction):
                # Don't tile stuff outside the main reduction loop
                disabled = node is DisableReduction
            elif disabled:
                pass
            else:
                yield node


class CantSplit(Exception):
    pass
