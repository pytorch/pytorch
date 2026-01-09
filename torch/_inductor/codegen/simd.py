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
import textwrap
from collections import Counter
from typing import Any, Generic, Optional, TYPE_CHECKING, Union
from typing_extensions import TypeVar

import sympy

import torch
import torch._logging
from torch._inductor import metrics
from torch._inductor.ir import MultiTemplateBuffer
from torch._inductor.tiling_utils import analyze_memory_coalescing
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.fx.immutable_collections import immutable_dict
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv, Identity, ModularIndexing
from torch.utils._sympy.symbol import (
    free_symbol_is_type,
    prefix_str,
    symbol_is_type,
    SymT,
)

from ..._dynamo.utils import counters
from .. import config, ir, scheduler
from ..analyze_preserves_zero_mask import prologue_preserves_zero_mask
from ..codecache import code_hash, PyCodeCache
from ..dependencies import MemoryDep, StarDep, WeakDep


if TYPE_CHECKING:
    from collections.abc import Callable

    from ..ir import IRNode

from ..optimize_indexing import indexing_dtype_strength_reduction
from ..runtime.coordinate_descent_tuner import CoordescTuner
from ..runtime.hints import DeviceProperties
from ..runtime.runtime_utils import green_text, last_power_of_2, yellow_text
from ..scheduler import BaseSchedulerNode, BaseScheduling, WhyNoFuse
from ..utils import (
    cache_property_on_self,
    expr_fits_within_32bit,
    get_dtype_size,
    IndentedBuffer,
    Placeholder,
    prefix_is_reduction,
    sympy_index_symbol,
    sympy_product,
    sympy_subs,
    unique,
)
from ..virtualized import ops, OpsWrapper, V
from .block_analysis import BlockPatternMatcher
from .common import CSEVariable, index_prevent_reordering, Kernel, PythonPrinter
from .multi_kernel import MultiKernel, SizeHintMultiKernel
from .simd_kernel_features import (
    DisableReduction,
    EnableReduction,
    NodeScheduleEntry,
    NodeScheduleMarker,
    SIMDKernelFeatures,
)


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from torch._inductor.tiling_utils import CoalesceVarAnalysis


log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")
schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")
fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")


pexpr = PythonPrinter().doprint

all_prefixes = OrderedSet(["z", "y", "x", "r0_", "r1_"])


def get_max_tiles(default: int = 2) -> int:
    max_tiles = torch._inductor.config.triton.max_tiles
    return max_tiles if max_tiles is not None else default


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
        var_list: list[sympy.Symbol],
        var_ranges: dict[sympy.Symbol, sympy.Expr],
        numel: sympy.Expr,
        prefix: str,
        *,
        kernel: SIMDKernel,
        divisor=sympy.S.One,
        length=sympy.S.One,
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

    @property
    @cache_property_on_self
    def is_reduction(self) -> bool:
        return prefix_is_reduction(self.prefix)

    def symbol(self) -> sympy.Symbol:
        return sympy_index_symbol(self.name)

    @property
    @cache_property_on_self
    def symt(self) -> SymT:
        prefix_to_symt = {prefix: symt for symt, prefix in prefix_str.items()}
        return prefix_to_symt[self.prefix]


class IterationRangesRoot(IterationRanges):
    """
    Root of a iteration range tree that represents a single
    tiled dimension in the output kernel. It contains multiple
    sets of iteration represented with IterationRangesEntry.
    """

    def __init__(
        self,
        name: str,
        numel: sympy.Expr,
        prefix: str,
        index: int,
        kernel: SIMDKernel,
        pid_cache: Optional[dict[str, str]] = None,
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
        self.nodes: dict[sympy.Expr, IterationRangesEntry] = {}
        # This is for re-ordering program ID in triton mm template
        # pid_cache["tl.program_id(0)"] = pid_m
        self.pid_cache: dict[str, str] = pid_cache

        # True if the dimension is implemented as a single program looping over
        # the full dimension (currently only used for non-persistent reduction)

        assert not is_loop or (self.is_reduction and grid_dim is None)
        self.is_loop = is_loop
        # Index of corresponding dimension on triton tensors
        self.tensor_dim = tensor_dim
        # Index of corresponding dimension in the triton grid
        self.grid_dim = grid_dim
        self.has_zdim = has_zdim

    def __repr__(self) -> str:
        return f"IterationRangesRoot({self.name!r}, {self.numel}, ...)"

    def cache_clear(self) -> None:
        for node in self.nodes.values():
            node.cache_clear()

    def index_sym(self) -> sympy.Symbol:
        return sympy_index_symbol(f"{self.prefix}index")

    def lookup(self, divisor: sympy.Expr, length: sympy.Expr) -> IterationRangesEntry:
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

    def construct_entries(
        self, lengths: list[sympy.Expr]
    ) -> list[IterationRangesEntry]:
        divisor = sympy.S.One
        itervars = []
        for length in reversed(lengths):
            itervars.append(self.lookup(divisor, length))
            divisor = divisor * length
        return [*reversed(itervars)]

    def construct(self, lengths: list[sympy.Expr]) -> list[sympy.Symbol]:
        return [e.symbol() for e in self.construct_entries(lengths)]

    def vars_and_sizes(
        self, index: sympy.Expr
    ) -> tuple[list[sympy.Symbol], list[sympy.Expr]]:
        """Figure out vars from this tree used in index"""

        def get_sort_key(x: IterationRangesEntry) -> tuple[int, bool]:
            """
            Gets the key for sorting nodes. When two nodes have the
            same divisor, the node with length as 1 should be handled
            first so the current divisor is not changed after multiplied
            node.length. Returns `not length_is_one_hint` for ascending
            sort.
            """
            divisor_hint = V.graph.sizevars.size_hint(
                x.divisor, fallback=config.unbacked_symint_fallback
            )
            length_is_one_hint = (
                V.graph.sizevars.size_hint(
                    x.length, fallback=config.unbacked_symint_fallback
                )
                == 1
            )
            return (divisor_hint, not length_is_one_hint)

        nodes = [V.kernel.range_tree_nodes.get(s) for s in index.free_symbols]
        nodes = [n for n in nodes if n and n.prefix == self.prefix]
        nodes.sort(key=lambda x: get_sort_key(x))
        divisor = sympy.S.One
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

        return [*reversed(index_vars)], [*reversed(sizes)]


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

    def set_name(self, name: str) -> None:
        self.codegen = lambda: name  # type: ignore[assignment]
        self.codegen.cache_clear = lambda: None  # type: ignore[method-assign]
        self.name = name

    def cache_clear(self) -> None:
        self.codegen.cache_clear()

    def _codegen(self) -> str:
        V.kernel.codegen_iteration_ranges_entry(self)
        return self.name

    def precomputed_args(self) -> list[sympy.Expr]:
        # for dynamic shapes, find parts of indexing expressions that have to be precomputed
        precomputed_args: list[sympy.Expr] = []
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

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, IterationRangesEntry)
        return self.name == other.name


def constant_repr(value: Union[int, float]) -> str:
    if value == float("inf"):
        return 'float("inf")'
    elif value == float("-inf"):
        return 'float("-inf")'
    elif math.isnan(value):
        return 'float("nan")'
    return repr(value)


CSEVariableType = TypeVar("CSEVariableType", bound=CSEVariable, default=CSEVariable)


@dataclasses.dataclass
class PartialAccumulate:
    buffer_name: str
    reduction_type: str
    value: Any


class SIMDKernel(Kernel[CSEVariableType], Generic[CSEVariableType]):
    """
    Common base class for Triton/Halide codegen which both use flattened indexing rather than loop nests.
    """

    sexpr: Callable[[sympy.Expr], str] = pexpr
    kexpr: Callable[[sympy.Expr], str]
    allow_block_ptr: bool = False
    # pyrefly: ignore [bad-override]
    kernel_name: str

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        features: SIMDKernelFeatures,
        pid_cache: Optional[dict[str, str]] = None,
        override_persistent_reduction: Optional[bool] = None,
        override_cooperative_reduction: Optional[bool] = None,
        tiling_scores: Optional[dict[str, sympy.Expr]] = None,
        mix_order_reduction: bool = False,
    ) -> None:
        if pid_cache is None:
            pid_cache = {}
        super().__init__()
        self.features = features
        self.mutations = features.get_mutations()
        self.body = IndentedBuffer()
        self.indexing_code = IndentedBuffer()
        self.numels = {
            prefix: V.graph.sizevars.simplify(val) for prefix, val in tiling.items()
        }
        self.range_trees: list[IterationRangesRoot] = []
        self.range_tree_nodes: dict[sympy.Symbol, IterationRangesEntry] = {}
        self.iter_vars_count = itertools.count()
        self.inside_reduction = features.is_reduction()
        self.cooperative_reduction: bool = (
            override_cooperative_reduction
            if override_cooperative_reduction is not None
            else self.should_use_cooperative_reduction()
        )
        self.tiling_scores: Optional[dict[str, sympy.Expr]] = tiling_scores
        self.tiling: dict[str, sympy.Expr] = tiling
        self.persistent_reduction: bool = (
            override_persistent_reduction
            if override_persistent_reduction is not None
            else self.should_use_persistent_reduction()
        )
        self.mix_order_reduction: bool = mix_order_reduction
        self.no_x_dim = self.want_no_x_dim()
        self.code_hash: Optional[str] = None
        # Info to enable multiple store_output calls for epilogue subtiling
        self.store_output_ctr = itertools.count()
        self.is_native_matmul = False
        if config.triton.native_matmul:
            for node in self.features.node_schedule:
                if (
                    isinstance(node, scheduler.SchedulerNode)
                    and isinstance(node.node, ir.ComputedBuffer)
                    and node.node.get_reduction_type() == "dot"
                ):
                    self.is_native_matmul = True
                    break

        # define this in a closure to make cache local to object
        @functools.cache
        def simplify_indexing(index: sympy.Expr):
            index = V.graph.sizevars.simplify_with_ranges(index, self.var_ranges())
            for tree in self.range_trees:
                index = self.combine_contiguous_dims(index, tree)

            return self.combine_modular_indexing_pairs(index)

        self.simplify_indexing = simplify_indexing
        self.initialize_range_tree(pid_cache)

        self.rsplit_size = 0
        self.saved_partial_accumulate: list[PartialAccumulate] = []

    def codegen_template_override(
        self,
        scheduling,
        template_node,
        epilogue_nodes,
        prologue_nodes,
        buf_name_to_prologue_group,
        prologue_preserves_zero_mask_fn,
        render,
        only_gen_src_code: bool,
    ) -> str | None:
        """Override template codegen. Return None to use default flow.

        External template handlers (e.g. Helion) can override this method
        to implement custom code generation.
        """
        return None

    def _get_store_output_subgraph_name(self, i: int) -> str:
        return f"<STORE_OUTPUT_{i}>"

    def get_store_output_count(self):
        total = next(self.store_output_ctr)
        self.store_output_ctr = itertools.count(start=total - 1, step=1)
        return total

    @property
    @cache_property_on_self
    def num_reduction_dims(self) -> int:
        return sum(prefix_is_reduction(prefix) for prefix in self.numels)

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        raise NotImplementedError

    def get_index_dtype_as_torch_dtype(self) -> torch.dtype:
        return self.features.select_index_dtype()

    @property
    def index_dtype(self) -> str:
        return self.dtype_to_str(self.get_index_dtype_as_torch_dtype())

    def want_no_x_dim(self) -> bool:
        return False

    def construct_range_trees(
        self,
        pid_cache: Optional[dict[str, str]],
        inside_reduction: bool,
        is_reduction: bool,
        numels: dict[str, sympy.Expr],
        no_x_dim: bool,
    ) -> list[IterationRangesRoot]:
        active_prefixes = OrderedSet(
            prefix for prefix in all_prefixes if prefix in numels
        )
        no_r_dim = not inside_reduction or not is_reduction

        def filtered_index_map(seq, mask) -> dict[Any, int]:
            return {
                val: idx for idx, val in enumerate(val for val in seq if val in mask)
            }

        grid_dims = ["x", "y", "z"]
        pointwise_tensor_dims = list(reversed(grid_dims))
        reduction_dims = ["r0_", "r1_"]
        if no_x_dim:
            tensor_dims = reduction_dims
        elif no_r_dim:
            tensor_dims = pointwise_tensor_dims
        else:
            tensor_dims = pointwise_tensor_dims + reduction_dims

        # Filter out unused tensor dims.
        # Convert to dicts for O(1) index lookup.
        tensor_dim_map = filtered_index_map(tensor_dims, active_prefixes)
        grid_dim_map = filtered_index_map(grid_dims, all_prefixes)

        range_trees = []
        for i, prefix in enumerate(active_prefixes):
            is_reduction = prefix_is_reduction(prefix)
            tensor_dim = tensor_dim_map.get(prefix)
            grid_dim = grid_dim_map.get(prefix)
            index = i if grid_dim is None else grid_dim
            range_trees.append(
                IterationRangesRoot(
                    f"{prefix}index",
                    numels[prefix],
                    prefix,
                    index,
                    self,  # type: ignore[arg-type]
                    pid_cache=pid_cache,
                    is_loop=is_reduction and not self.persistent_reduction,
                    tensor_dim=tensor_dim,
                    grid_dim=grid_dim,
                    has_zdim="z" in numels,
                )
            )
        return range_trees

    def initialize_range_tree(self, pid_cache: dict[str, str]) -> None:
        range_trees = self.construct_range_trees(
            pid_cache,
            self.inside_reduction,
            self.features.is_reduction(),
            self.numels,
            self.no_x_dim,
        )
        self.range_trees.extend(range_trees)

    def finalize_indexing(self, indices: Sequence[sympy.Expr]) -> None:
        """
        Hook called right before codegen with every index that will be
        used in the fused kernel.
        """

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
        prior = self.inside_reduction
        self.inside_reduction = False
        try:
            return self.store(name, index, value)
        finally:
            self.inside_reduction = prior

    def should_use_cooperative_reduction(self) -> bool:
        return False  # defined in subclass

    def should_use_persistent_reduction(self) -> bool:
        return False  # defined in subclass

    def var_ranges(self) -> dict[sympy.Symbol, sympy.Expr]:
        return dict(
            itertools.chain.from_iterable(
                tree.var_ranges.items() for tree in self.range_trees
            )
        )

    def triton_tensor_ndim(self) -> int:
        return sum(int(tree.tensor_dim is not None) for tree in self.range_trees)

    def indexing_size_str(self, i: int) -> str:
        sizes = ["None"] * self.triton_tensor_ndim()
        sizes[i] = ":"
        return f"[{', '.join(sizes)}]"

    def dense_size_list(self) -> list[str]:
        sizes = ["1"] * self.triton_tensor_ndim()
        for tree in self.range_trees:
            if tree.tensor_dim is None:
                continue

            if not tree.is_reduction or self.inside_reduction:
                sizes[tree.tensor_dim] = f"{tree.prefix.upper()}BLOCK"
        return sizes

    def create_constant_mask(self, entry) -> str:
        x = entry.prefix
        if entry.tensor_dim is None:
            sizestr = self.dense_size_str()
            return f"{x}mask = tl.full({sizestr}, True, tl.int1)"
        sizes = ["None"] * self.triton_tensor_ndim()
        sizes[entry.tensor_dim] = ":"
        suffix = ", ".join(sizes)
        out = f"{x}mask = tl.full([{x.upper()}BLOCK], True, tl.int1)[{suffix}]"
        return out

    def dense_size_str(self) -> str:
        sizes = self.dense_size_list()
        return f"[{', '.join(sizes)}]"

    def combine_modular_indexing_pairs(self, index: sympy.Expr) -> sympy.Expr:
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
                    sympy.S.One, tree_node.root.numel
                ).symbol()
            },
        )

    def combine_contiguous_dims(
        self, index: sympy.Expr, tree: IterationRangesRoot
    ) -> sympy.Expr:
        if expand_res := V.graph.sizevars.expand_floor_div(index):
            new_index, denominator = expand_res  # type: ignore[misc]
            return FloorDiv(self._combine_contiguous_dims(new_index, tree), denominator)
        else:
            return self._combine_contiguous_dims(index, tree)

    def _combine_contiguous_dims(
        self, index: sympy.Expr, tree: IterationRangesRoot
    ) -> sympy.Expr:
        """
        More aggressive simplification to merge contiguous dims
        """
        if isinstance(index, (sympy.Integer, sympy.Symbol)):
            return index
        index_vars, sizes = tree.vars_and_sizes(index)
        if len(sizes) <= 1:
            return index
        new_sizes, reindex, _prune = V.graph.sizevars._simplify_loops(
            index_vars, sizes, index_prevent_reordering([index], index_vars, sizes)
        )
        if new_sizes == sizes:
            return index
        new_index_vars = tree.construct(new_sizes)
        new_index = sympy_subs(index, dict(zip(index_vars, reindex(new_index_vars))))
        return new_index

    def disable_reduction(self) -> contextlib.AbstractContextManager[None]:
        should_flush = self.range_trees[-1].is_loop or self.cooperative_reduction

        @contextlib.contextmanager
        def ctx():
            if not self.features.is_reduction():
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

    def set_ranges(self, *lengths: sympy.Expr) -> list[sympy.Symbol]:
        assert len(lengths) == len(self.range_trees)
        return [
            ranges.construct(length)
            for length, ranges in zip(lengths, self.range_trees)
        ]

    @staticmethod
    def _split_iteration_ranges(
        groups: Iterable[sympy.Expr], lengths: Sequence[Sequence[sympy.Expr]]
    ) -> tuple[
        list[list[sympy.Expr]], list[list[Callable[[list[sympy.Expr]], sympy.Expr]]]
    ]:
        # Special case: if a node's sizes are ([], []), there's nothing to split.
        if all(len(length) == 0 for length in lengths):
            return [[] for group in groups], []

        sv = V.graph.sizevars
        new_ranges: list[list[sympy.Expr]] = [[] for _ in groups]
        remaining = [sv.simplify(g) for g in groups]
        var_count = itertools.count()

        def add_range(i: int, expr: sympy.Expr) -> int:
            expr = sv.simplify(expr)
            if not sv.statically_known_multiple_of(remaining[i], expr):
                raise CantSplit(remaining[i], expr)
            # guard on the last item out
            remaining[i] = FloorDiv(remaining[i], expr)
            new_ranges[i].append(expr)
            return next(var_count)

        def make_combined(
            sizes: list[sympy.Expr], idxs: list[int]
        ) -> Callable[[list[sympy.Expr]], sympy.Expr]:
            """
            Builds the nested expression:
              ((...((s1*v[i1] + v[i2]) * s2 + v[i3]) ... ) * sk + v[i(k+1)])
            """
            assert len(idxs) == len(sizes) + 1

            def getter(flat_vars: list[sympy.Expr]) -> sympy.Expr:
                expr = flat_vars[idxs[0]]
                for s, idx in zip(sizes, idxs[1:]):
                    expr = s * expr + flat_vars[idx]
                return expr

            return getter

        return_getters_groups = []
        current_group = 0
        for length_group in lengths:
            return_getters = []
            for size in length_group:
                if sv.statically_known_equals(size, 1):  # type: ignore[arg-type]
                    return_getters.append(lambda _: sympy.S.Zero)
                    continue

                while current_group < len(remaining) and sv.statically_known_equals(
                    remaining[current_group],
                    1,  # type: ignore[arg-type]
                ):
                    # scroll to next group with remaining elements
                    current_group += 1

                # During native matmul on bmm, we enforce tiling order (z, y, x, r).
                # When fusing a bmm node with loop (z, y, x, r) with a pw node
                # of shape (z*y*x, 1), we need to split the pw iteration range
                # into three dimensions.
                # The group becomes [z, y, x, 1], with lengths ([z*y*x], []).
                # In this case, we decompose the combined size z*y*x into three
                # consecutive groups. Previously, _split_iteration_ranges supported
                # splitting into at most two dimensions, but we now extend it to do
                # three splits when the total size is divisible by all three.

                # is group having (z,y,x,r=1) form?
                is_bmm_then_pw = len(remaining) == 4 and remaining[-1] == 1
                if (
                    current_group + 2 < len(remaining)
                    and sv.statically_known_gt(
                        size, remaining[current_group] * remaining[current_group + 1]
                    )
                    and is_bmm_then_pw
                ):
                    # need to break size in three
                    if not sv.statically_known_multiple_of(
                        size, remaining[current_group] * remaining[current_group + 1]
                    ):
                        raise CantSplit

                    size1 = remaining[current_group]
                    size2 = remaining[current_group + 1]
                    size3 = FloorDiv(size, size1 * size2)
                    return_getters.append(
                        make_combined(
                            [size2, size3],
                            [
                                add_range(current_group, size1),
                                add_range(current_group + 1, size2),
                                add_range(current_group + 2, size3),
                            ],
                        )
                    )

                # Two-dimensional tiling: split size across current_group and next group.
                elif current_group + 1 < len(remaining) and (
                    sv.statically_known_gt(size, remaining[current_group])
                    or
                    # statically_known_gt(size, remaining) may return False for symbolic
                    # expressions like 64*u0 vs u0, because both could be 0. Similarly for
                    # backed expressions like s25*(((s70 - 5)//4)) - s25 and
                    # (s25*(((s70 - 5)//4)) - s25)*64.
                    # We want to assume tensor sizes are not 0 and pass the gt
                    # using the following logic.
                    #
                    # if A//B = C and C >= 1
                    # then A = B * C + R
                    # and assuming A!=0
                    # A must be > B .
                    #
                    sv.statically_known_gt(FloorDiv(size, remaining[current_group]), 1)
                ):
                    # need to break size in two
                    if not sv.statically_known_multiple_of(
                        size, remaining[current_group]
                    ):
                        raise CantSplit(size, remaining[current_group])

                    size1 = remaining[current_group]
                    size2 = FloorDiv(size, remaining[current_group])
                    return_getters.append(
                        make_combined(
                            [size2],
                            [
                                add_range(current_group, size1),
                                add_range(current_group + 1, size2),
                            ],
                        )
                    )
                else:
                    if current_group < len(remaining):
                        return_getters.append(
                            operator.itemgetter(add_range(current_group, size))
                        )
            return_getters_groups.append(return_getters)

        assert all(V.graph.sizevars.size_hint(s) == 1 for s in remaining), (
            f"failed to set ranges {remaining} {lengths}"
        )
        return new_ranges, return_getters_groups

    @classmethod
    def prepare_split_iteration_lengths(
        cls,
        groups: Iterable[sympy.Expr],
        lengths: Sequence[Sequence[sympy.Expr]],
        reduction_numel: sympy.Expr = sympy.S.One,
    ) -> Sequence[Sequence[sympy.Expr]]:
        "Fill in the reduction numel of lengths if missing"
        sizevars = V.graph.sizevars
        if len(lengths[1]) == 0 and (
            not sizevars.statically_known_equals(reduction_numel, sympy.S.One)
            and sizevars.statically_known_equals(
                sympy_product(groups),
                sympy_product(lengths[0]) * reduction_numel,
            )
        ):
            return (lengths[0], [reduction_numel])

        return lengths

    @classmethod
    def is_compatible(
        cls,
        groups: Iterable[sympy.Expr],
        lengths: Sequence[Sequence[sympy.Expr]],
        reduction_numel: sympy.Expr = sympy.S.One,
    ) -> bool:
        lengths = cls.prepare_split_iteration_lengths(groups, lengths, reduction_numel)

        try:
            cls._split_iteration_ranges(groups, lengths)
            return True
        except CantSplit:
            return False

    def split_and_set_ranges(
        self, lengths: Sequence[Sequence[sympy.Expr]]
    ) -> list[list[sympy.Expr]]:
        """
        Split and set iteration ranges for the kernel based on the provided lengths.

        This method maps the kernel's tiling structure to the node's iteration space,
        handling both pointwise and reduction dimensions appropriately.

        Args:
            lengths: A sequence of sequences of symbolic expressions representing
                    the sizes of different dimensions for each node.

        Returns:
            A list of lists of symbolic expressions representing the mapped
            iteration variables for each dimension.
        """
        # Create a dictionary mapping each range tree prefix to its total number of elements
        tiling = {rt.prefix: rt.numel for rt in self.range_trees}

        # If we're not inside a reduction loop, set all reduction dimensions to 1
        # This effectively disables reduction dimensions when not needed
        if not self.inside_reduction:
            for prefix in tiling:
                if prefix_is_reduction(prefix):
                    tiling[prefix] = sympy.S.One

        # Extract the values from the tiling dictionary to create groups
        groups = [*tiling.values()]

        # Map the kernel's group structure to the node's sizes and set the ranges
        # using the set_ranges method, returning the resulting iteration variables
        return self.map_kernel_groups_to_node_sizes(groups, lengths, self.set_ranges)

    @classmethod
    def map_kernel_groups_to_node_sizes(
        cls,
        groups: Sequence[sympy.Expr],
        lengths: Sequence[Sequence[sympy.Expr]],
        set_ranges,
    ) -> list[list[sympy.Expr]]:
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
        if len(lengths) == len(groups) and all(
            V.graph.sizevars.simplify(sympy_product(x) - g) == 0
            for x, g in zip(lengths, groups)
        ):
            return set_ranges(*lengths)

        new_ranges, return_getters_groups = cls._split_iteration_ranges(groups, lengths)
        itervars = [*itertools.chain.from_iterable(set_ranges(*new_ranges))]
        return [[fn(itervars) for fn in fns] for fns in return_getters_groups]

    def is_indirect_indexing(self, index: sympy.Expr) -> bool:
        # tmpX  means indirect indexing
        return free_symbol_is_type(index, SymT.TMP)

    def is_broadcasted(self, index: sympy.Expr) -> bool:
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
            for idx_range, iter_range in zip(index_numels, self.numels.values())
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
    ) -> sympy.Expr:
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

    def active_range_trees(self) -> list[IterationRangesRoot]:
        return [
            t for t in self.range_trees if not t.is_reduction or self.inside_reduction
        ]

    def codegen_indexing(self, expr: sympy.Expr) -> sympy.Expr:
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
                        self.range_tree_nodes[sym].expr,
                        replacements,  # type: ignore[index]
                    )
                self.range_tree_nodes[sym].codegen()  # type: ignore[index]
        return expr

    def codegen_nan_check(self) -> None:
        raise NotImplementedError("NYI: codegen_nan_check")

    def deallocate_workspaces(self):
        wrapper = V.graph.wrapper_code
        for ws in reversed(self.args.workspace_args):
            wrapper.generate_workspace_deallocation(ws)

    def call_kernel(
        self, name: str, node: Optional[IRNode] = None, deallocate_ws: bool = True
    ) -> None:
        raise NotImplementedError("NYI: call_kernel")

    @contextlib.contextmanager
    def mask_loads(
        self, mask: Union[str, OpsWrapper], value: Union[int, float]
    ) -> Iterator[str]:
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

    def get_strides_of_load(self, index: sympy.Expr) -> dict[sympy.Symbol, sympy.Expr]:
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

    def estimate_flops(self) -> Optional[int]:
        flops = [
            node.estimate_flops()
            for node in NodeScheduleMarker.only_nodes(self.features.node_schedule)
        ]
        return sum(filter(None, flops))

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
        buf_accesses = self.features.buf_accesses()

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
        out_numel = V.graph.sizevars.size_hint(
            sympy_product(self.numels.values()),
            fallback=config.unbacked_symint_fallback,
        )
        for i, arg in enumerate(call_args):
            # "buf" may be narrowed. In this case, the number of memory accesses
            # should be estimated based on the reinterpreted layout.
            # On the other hand, buf may be broadcasted. In this case,
            # counting the size of the underline storage would give us
            # a better estimation in terms of memory accesses.
            if arg not in buf_accesses:
                nbytes.append(0)
                continue
            arg_numel = V.graph.get_numel(arg)
            buf_size = V.graph.sizevars.size_hint(
                arg_numel, fallback=config.unbacked_symint_fallback
            )
            if buf_size > out_numel:
                # This arg points to a buf that has been sliced.
                # We need to count each individual slice to have
                # a better estimation.
                indices = OrderedSet[Any]()
                no_index_dep_count = 0
                for dep in buf_accesses[arg]:
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
            # pyrefly: ignore [bad-argument-type]
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

        argdefs, call_args, _signature, _ = self.args.python_argdefs()
        uniform_stride_order = None
        # pyrefly: ignore [bad-assignment]
        for arg_name in call_args:
            buf = V.graph.try_get_buffer(arg_name)
            if not buf:
                continue
            layout = buf.get_layout()
            if len(layout.size) == 4:
                # ignore the tensor if only 1 dimension is non-zero
                if len([x for x in layout.size if x == 1]) == 3:
                    continue
                stride_order = ir.get_stride_order(layout.stride)
                if uniform_stride_order is None:
                    uniform_stride_order = stride_order
                elif uniform_stride_order != stride_order:
                    msg = yellow_text(
                        f"Expected stride order {uniform_stride_order}, but found stride order"
                        + f" {stride_order} for kernel {kernel_name}"
                    )
                    log.warning(msg)

                    stride_order_list = [
                        ir.get_stride_order(
                            V.graph.get_buffer(name).get_layout().stride
                        )
                        if V.graph.try_get_buffer(name)
                        else None
                        for name in call_args
                    ]
                    size_list = [
                        V.graph.get_buffer(name).get_layout().size
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

                    argdef_names = [x.name for x in argdefs]
                    msg = yellow_text(
                        f"  param names {argdef_names}\n  buf names {call_args}\n  strides {stride_order_list}"
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
        rnumel = ops.index_expr(self.features.reduction_numel, dtype)
        mean = ops.truediv(sum_, rnumel)

        self.inside_reduction = True
        dx = ops.sub(value, mean)
        dx2 = ops.mul(dx, dx)
        m2 = ops.reduction(dtype, dtype, "sum", dx2)
        return OpsWrapper._unwrap((mean, m2, rnumel))

    def prepare_softmax_twopass_fallback(self, dtype, value):
        vmax = ops.reduction(dtype, dtype, "max", value)
        sub = ops.sub(value, vmax)
        exp = ops.exp(sub)
        vsum = ops.reduction(dtype, dtype, "sum", exp)
        return OpsWrapper._unwrap((vmax, vsum))

    def codegen_kernel(self):
        raise NotImplementedError

    def codegen_body(self):
        pass

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry):
        pass


class SIMDScheduling(BaseScheduling):
    """
    Single Instruction Multiple Data parent class used for fusion across
    multiple different backends.
    """

    kernel_type: type[Any] = SIMDKernel  # override in subclass

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
                from torch._inductor.scheduler import MixOrderReduction

                reduction_can_fuse = MixOrderReduction.can_fuse(node1, node2)

            if not reduction_can_fuse:
                why(
                    "numel/rnumel mismatch (reduce) (%s, %s), (%s, %s)",
                    numel1,
                    numel2,
                    rnumel1,
                    rnumel2,
                )

            if reduction_can_fuse and (
                node1.is_native_matmul() or node2.is_native_matmul()
            ):
                # Ensure node1 is always the native matmul side
                if not node1.is_native_matmul():
                    node1, node2 = node2, node1

                # 1. A native matmul node keeps its original loop order.
                #    For example: C[z,y,x] = torch.bmm(A[z,y,r], B[z,r,x]) keeps (z,y,x) order.
                #    (see simplify_and_reorder in ir.py)
                #
                # 2. Triton kernels with native matmul always tile loops as (z,y,x)
                #    (see get_tiling_and_scores in this file)
                #
                # 3. If a candidate node (node2) uses a different loop order (e.g., (z,x,y,r)),
                #    its tiling is incompatible with native matmul tiling (z,y,x,r).
                #    This means _split_iteration_ranges will fail, so these nodes should not be fused.
                tiling = self.select_tiling(node1.get_nodes(), numel1, rnumel1)
                if not all(
                    SIMDKernel.is_compatible(
                        tiling.values(), n2.get_ranges(), reduction_numel=rnumel1
                    )
                    for n2 in node2.get_nodes()
                ):
                    why("invalid loop order and tiling for native matmul")
                    return False

            return reduction_can_fuse

        if not node1.is_reduction() and not node2.is_reduction():
            if not (numel1 == numel2 and rnumel1 == rnumel2):
                if not node2.is_template():
                    why(
                        "numel/rnumel mismatch (non-reduce) (%s, %s), (%s, %s)",
                        numel1,
                        numel2,
                        rnumel1,
                        rnumel2,
                    )
                    return False
                else:
                    # prologue fusion input sizes differ from output group
                    # fuse so long as this node matches the group of existing prologue nodes
                    for node in node2.get_nodes():
                        # dont need to check epilogue nodes for prologue fusion, break after template
                        if node.is_template():
                            break
                        # we would have already restricted prologue from fusing if it had multiple
                        # uses, so it must be fusing into this node
                        if not node.used_buffer_names() & node1.get_buffer_names():
                            continue
                        _, (pro_numel, pro_rnumel) = node.group
                        if not (numel1 == pro_numel and rnumel1 == pro_rnumel):
                            why(
                                "numel/rnumel mismatch prologue mismatch (%s, %s), (%s, %s)",
                                numel1,
                                pro_numel,
                                rnumel1,
                                pro_rnumel,
                            )
                            return False

            for n in (node1, node2):
                if n.is_template():
                    return True

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
                    is_reduction_tiling_valid = tuple(
                        self.select_tiling(node1.get_nodes(), numel1).values()
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
        node_schedule: list[Any] = []
        done = OrderedSet[scheduler.BaseSchedulerNode]()
        # Writes with a reduced shape, meaning they are only present once the
        # reduction loop has ended
        not_ready_yet_nodes: OrderedSet[str] = OrderedSet()
        current_loop_buffer_usage: OrderedSet[str] = OrderedSet()
        maybe_split_index: Optional[int] = None

        def fits_in_main_body(n):
            _, (node_numel, node_rnumel) = n.group
            return (node_numel == numel and node_rnumel == rnumel) or (
                node_numel == numel * rnumel and node_rnumel == 1
            )

        def fits_outside_reduction(n):
            _, (node_numel, node_rnumel) = n.group
            return node_numel == numel and node_rnumel == 1 and rnumel != 1

        def expect_improved_memory_usage(n):
            for read in n.read_writes.reads:
                if read.name in current_loop_buffer_usage:
                    return True
            return False

        def schedule_node_in_loop(n):
            done.add(n)
            node_schedule.append(n)
            current_loop_buffer_usage.update([x.name for x in n.read_writes.reads])

            # A scan is modelled as a reduction in the scheduler but has a
            # full sized output that can be used inside the loop body
            if (
                n.is_reduction()
                and isinstance(n, scheduler.SchedulerNode)
                and isinstance(n.node, ir.ComputedBuffer)
                and not isinstance(n.node.data, ir.Scan)
            ):
                not_ready_yet_nodes.add(n.get_name())
            else:  # this node is available within the loop
                current_loop_buffer_usage.update([x.name for x in n.read_writes.writes])

        @contextlib.contextmanager
        def end_current_reduction_loop():
            nonlocal maybe_split_index
            if node_schedule and node_schedule[-1] is EnableReduction:
                node_schedule.pop()
            else:
                node_schedule.append(DisableReduction)
            if maybe_split_index:
                node_schedule.insert(maybe_split_index, DisableReduction)
                node_schedule.insert(maybe_split_index + 1, EnableReduction)
                maybe_split_index = None
            yield
            node_schedule.append(EnableReduction)
            not_ready_yet_nodes.clear()
            current_loop_buffer_usage.clear()

        def requires_closing_previous_reduction(node, node_schedule):
            if rnumel == 1:
                return False
            if not not_ready_yet_nodes & node.ancestors:
                return False
            assert node_schedule and not isinstance(
                node_schedule[-1], (EnableReduction, DisableReduction)
            )
            return bool(not_ready_yet_nodes)

        for node in nodes:
            if node in done:
                continue
            done.add(node)

            if fits_in_main_body(node):
                if requires_closing_previous_reduction(node, node_schedule):
                    with end_current_reduction_loop():
                        pass  # need to start a new reduction loop

                if current_loop_buffer_usage and not expect_improved_memory_usage(node):
                    # If we don't improve memory usage, then it is better to split into two loops
                    maybe_split_index = maybe_split_index or len(node_schedule)
                else:
                    # Memory usage got improved, cancel the loop split
                    maybe_split_index = None

                schedule_node_in_loop(node)
            elif fits_outside_reduction(node):
                with end_current_reduction_loop():
                    node_schedule.append(node)
            else:
                raise NotImplementedError(
                    f"unexpected group: ({numel}, {rnumel}) != {node.group[1]}"
                )

        return node_schedule

    def codegen_mix_order_reduction(self, node):
        node1, node2 = node.node1, node.node2

        # Make sure there are no producer/consumer relationship
        assert not (node1.ancestors & node2.get_operation_names()) and not (
            node2.ancestors & node1.get_operation_names()
        )

        self._codegen_mix_order_reduction(node1, node2)

    def _split_mix_order_reduction_epilogue(self, node):
        # TODO: do more validation here
        nodes = node.get_nodes()
        reductions = []
        epilogues = []
        for node in nodes:
            if node.is_reduction():
                reductions.append(node)
            else:
                epilogues.append(node)
        return reductions, epilogues

    def _generate_kernel_code_for_mix_order_reduction(
        self, kernel_features, split_size, for_benchmark
    ):
        """
        for_benchmark:
            True if the generated code is for benchmarking. We need make
            sure benchmark harness code is generated.
        """
        numel, rnumel = kernel_features.numel, kernel_features.reduction_numel
        node_schedule = kernel_features.node_schedule

        kernel = self.create_kernel_choices(
            kernel_features,
            [{"x": numel, "r0_": rnumel}],
            {
                "features": kernel_features,
                "tiling_scores": None,
                "mix_order_reduction": True,
                "override_persistent_reduction": True,
            },
        )[0]
        assert kernel.persistent_reduction
        assert kernel.mix_order_reduction
        kernel.rsplit_size = split_size
        self.codegen_node_schedule_with_kernel(node_schedule, kernel)

        # allocate workspace for this kernel
        _, ws_name, ws_off = kernel.args.workspace(
            len(kernel.saved_partial_accumulate)
            * kernel.numels["r0_"]
            * ((kernel.numels["x"] + kernel.rsplit_size - 1) // kernel.rsplit_size),
            False,
            dtype=torch.float,
        )
        assert ws_off == 0, f"{ws_off=}"
        with kernel:
            kernel.codegen_body()

        stack = contextlib.ExitStack()
        with V.set_kernel_handler(kernel), stack:
            if for_benchmark:
                stack.enter_context(config.patch(benchmark_kernel=True))
            src_code = kernel.codegen_kernel()

        if for_benchmark:
            # only do this if we are doing benchmarking.
            # When we are generating final code, the kernel name
            # should be decided differently with node type, fx node name
            # etc.
            src_code = src_code.replace(str(Placeholder.KERNEL_NAME), "triton_")
        return kernel, ws_name, src_code

    def benchmark_codegened_module(
        self, mod, n_spills_threshold=8, node_names: Optional[OrderedSet[str]] = None
    ) -> tuple[float, str]:
        raise NotImplementedError

    def _codegen_mix_order_reduction(self, node1, node2):
        numel, rnumel = scheduler.MixOrderReduction.get_numel_rnumel(node1)

        if not V.graph.sizevars.evaluate_expr(sympy.Gt(numel, rnumel)):
            return self._codegen_mix_order_reduction(node2, node1)

        def _pick_split_size():
            # the overridden has highest priority
            if config.triton.mix_order_reduction_split_size is not None:
                return config.triton.mix_order_reduction_split_size

            # heuristics based on number of SMs
            device_prop = DeviceProperties.create(node1.get_device())
            num_sm = device_prop.multi_processor_count
            estimated_num_splits = num_sm * 8

            # split_size is decided based on hint
            numel_hint = V.graph.sizevars.size_hint(numel)
            split_size = max(last_power_of_2(numel_hint // estimated_num_splits), 16)
            split_size = min(split_size, 128)
            return split_size

        split_size = _pick_split_size()

        # pyrefly: ignore [bad-assignment]
        metrics.codegen_mix_order_reduction += 1

        assert V.graph.sizevars.evaluate_expr(sympy.Gt(numel, rnumel))

        # split epilogue out of node2
        node2_reductions, node2_epilogue = self._split_mix_order_reduction_epilogue(
            node2
        )

        converted_nodes = []
        for subnode in node2_reductions:
            subnode.cancel_reduction_split()
            converted = subnode.extract_pw_from_reduction()
            converted.swap_pw_red_dimension()
            converted_nodes.append(converted)
        node_schedule = self.generate_node_schedule(
            node1.get_nodes() + converted_nodes, numel, rnumel
        )
        kernel_features = SIMDKernelFeatures(node_schedule, numel, rnumel)

        # The autotuning is skipped in deterministic mode
        if (
            not torch._inductor.config.deterministic
            and config.triton.mix_order_reduction_split_size is None
            and (
                config.triton.mix_order_reduction_autotune_split_size
                or config.max_autotune
                or config.coordinate_descent_tuning
            )
        ):

            def _bench(candidate_split_size):
                _, _, src_code = self._generate_kernel_code_for_mix_order_reduction(
                    kernel_features,
                    split_size=candidate_split_size,
                    for_benchmark=True,
                )
                mod = PyCodeCache.load(src_code)
                ms, _ = self.benchmark_codegened_module(mod)
                return ms

            split_size = CoordescTuner.autotune_single_field(
                _bench,
                split_size,
                8,
            )

        kernel, ws_name, src_code = self._generate_kernel_code_for_mix_order_reduction(
            kernel_features,
            split_size=split_size,
            for_benchmark=False,
        )

        # rename intermediate reduction output to final reduction
        # output
        is_split_reduction = bool(node2_reductions[0].node._split_size)
        rename = {}
        if is_split_reduction:
            for subnode in node2_reductions:
                bufname = subnode.get_outputs()[0].node.get_name()
                username = (
                    subnode.get_outputs()[0]
                    .users[0]
                    .node.get_outputs()[0]
                    .node.get_name()
                )
                rename[bufname] = username
                assert self.scheduler
                self.scheduler.removed_ops.add(
                    subnode.get_outputs()[0].users[0].node.get_name()
                )
                V.graph.removed_buffers.add(bufname)

            for partial_accum in kernel.saved_partial_accumulate:
                partial_accum.buffer_name = rename.get(
                    partial_accum.buffer_name, partial_accum.buffer_name
                )

        kernel_name = self.define_kernel(src_code, node_schedule, kernel)
        kernel.kernel_name = kernel_name
        kernel.code_hash = code_hash(src_code)

        with V.set_kernel_handler(kernel):
            for node in kernel_features.scheduler_nodes():
                # No need to allocate buffer for split reduction
                # since we are gonna to allocate workspace to store the
                # intermediate reduction reduction
                if node.get_outputs()[0].node.get_name() not in rename:
                    node.mark_run()

        V.graph.wrapper_code.make_comment("# Call mix order reduction kernel")
        self.codegen_comment(node_schedule, None)
        # workspace args is still needed after the call
        kernel.call_kernel(kernel.kernel_name, deallocate_ws=False)
        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove

        # a extra round of reduction
        assert len(converted_nodes) == len(kernel.saved_partial_accumulate)
        nsplit = V.graph.wrapper_code.codegen_python_sizevar(
            (numel + split_size - 1) // split_size
        )
        for idx, partial_accum in enumerate(kernel.saved_partial_accumulate):
            buffer_name = partial_accum.buffer_name

            stride_str = f"{nsplit} * {rnumel}"
            start = f"{idx} * {stride_str}"
            end = f"({idx} + 1) * {stride_str}"
            reduction_type2op = {
                "min": "amin",
                "max": "amax",
            }
            opname = reduction_type2op.get(
                partial_accum.reduction_type, partial_accum.reduction_type
            )

            final_reduce = f"{buffer_name} = {ws_name}[{start} : {end}].view({nsplit}, {rnumel}).{opname}(dim=0)"
            # The workspace tensor is in torch.float, need a cast if the buffer is
            # not.
            if (buffer_dtype := V.graph.get_dtype(buffer_name)) != torch.float:
                final_reduce += f".to({buffer_dtype})"
            V.graph.wrapper_code.writeline(final_reduce)
            # mark the buffer as allocated, so we don't try to allocate
            # it again when it's later used
            V.graph.wrapper_code.allocated.add(buffer_name)

        kernel.deallocate_workspaces()

        if node2_epilogue:
            self._codegen_nodes(node2_epilogue)

        self.free_buffers_in_scheduler()

    def _codegen_nodes(
        self,
        nodes: Sequence[scheduler.SchedulerNode],
        coalesce_analysis: Optional[CoalesceVarAnalysis] = None,
    ):
        assert self.scheduler
        nodes = [
            node for node in nodes if node.get_name() not in self.scheduler.removed_ops
        ]
        if not nodes:
            return
        _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group

        node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
        schedule_log.debug("Schedule:\n %s", node_schedule)

        return self.codegen_node_schedule(
            SIMDKernelFeatures(node_schedule, numel, rnumel, coalesce_analysis)
        )

    def codegen_node(
        self, node: Union[scheduler.FusedSchedulerNode, scheduler.SchedulerNode]
    ):
        """
        Given a set of pre-fused nodes, generate a Triton kernel.
        """
        assert self.scheduler
        nodes = [
            node
            for node in node.get_nodes()
            if node.get_name() not in self.scheduler.removed_ops
        ]
        if len(nodes) == 0:
            return

        if torch._inductor.config.triton.coalesce_tiling_analysis:
            if len(nodes) != len(node.get_nodes()):
                assert self.scheduler
                node = scheduler.FusedSchedulerNode(self.scheduler, nodes)
            coalesce_analysis = analyze_memory_coalescing(node)
        else:
            coalesce_analysis = None

        return self._codegen_nodes(nodes, coalesce_analysis)  # type: ignore[arg-type]

    @staticmethod
    def can_use_32bit_indexing(
        numel: sympy.Expr,
        buffers: Iterable[
            Union[ir.Buffer, ir.TensorBox, ir.TorchBindObject, ir.IRNode]
        ],
    ) -> bool:
        int_max = torch.iinfo(torch.int32).max

        if not expr_fits_within_32bit(numel):
            return False

        # Any use of a MultiOutputLayout will create a buffer with a
        # Layout whose sizes are accounted for
        buf_sizes = [
            buf.get_layout().storage_size()
            for buf in buffers
            if buf.has_tensor_output()
        ]

        for buf in buffers:
            if not buf.has_tensor_output() and isinstance(buf, ir.MutationOutput):
                mutated_bufs = buf.get_mutation_buffers()
                buf_sizes += [
                    buf.get_layout().storage_size()
                    for buf in mutated_bufs
                    if buf.has_tensor_output()
                ]

        if not all(expr_fits_within_32bit(size) for size in buf_sizes):
            return False

        # Only install guards for 32-bit indexing as there is no correctness
        # issue with using 64-bit for everything
        V.graph.sizevars.check_leq(numel, int_max)  # type: ignore[arg-type]
        for size in buf_sizes:
            V.graph.sizevars.check_leq(size, int_max)  # type: ignore[arg-type]
        return True

    def codegen_node_schedule(self, kernel_features: SIMDKernelFeatures):
        """
        Generate code for nodes in kernel_features
        """
        node_schedule = kernel_features.node_schedule

        tiling, tiling_score = self.get_tiling_and_scores(
            node_schedule,
            kernel_features.numel,
            kernel_features.reduction_numel,
            kernel_features.coalesce_analysis,
        )
        kernels = self.create_kernel_choices(
            kernel_features,
            [tiling],
            {"features": kernel_features, "tiling_scores": tiling_score},
        )
        for kernel in kernels:
            self.codegen_node_schedule_with_kernel(node_schedule, kernel)
        MultiKernel.merge_workspaces_inplace(kernels)
        for kernel in kernels:
            with V.set_kernel_handler(kernel):
                src_code = kernel.codegen_kernel()
            kernel_name = self.define_kernel(src_code, node_schedule, kernel)
            log.debug("Generating kernel code with kernel_name: %s", kernel_name)
            kernel.kernel_name = kernel_name
            kernel.code_hash = code_hash(src_code)
        del kernel

        final_kernel: Union[SIMDKernel, MultiKernel]
        if len(kernels) > 1:
            final_kernel = MultiKernel(kernels)
        else:
            (final_kernel,) = kernels

        with V.set_kernel_handler(final_kernel):
            for node in kernel_features.scheduler_nodes():
                node.mark_run()

        # filter out NodeScheduleMarker
        base_scheduler_nodes = [
            node for node in node_schedule if isinstance(node, BaseSchedulerNode)
        ]
        self.codegen_comment(base_scheduler_nodes, final_kernel.kernel_name)
        if config.cpp.enable_kernel_profile:
            V.graph.wrapper_code.write_kernel_context_guard_begin()
            V.graph.wrapper_code.write_kernel_context_guard(
                final_kernel.kernel_name,
                base_scheduler_nodes,  # type: ignore[arg-type]
            )
        final_kernel.call_kernel(final_kernel.kernel_name)
        if config.cpp.enable_kernel_profile:
            V.graph.wrapper_code.write_kernel_context_guard_end()

        if config.nan_asserts:
            final_kernel.codegen_nan_check()
        if config.warn_mix_layout:
            final_kernel.warn_mix_layout(kernels[0].kernel_name)

        V.graph.removed_buffers |= final_kernel.removed_buffers
        V.graph.inplaced_to_remove |= final_kernel.inplaced_to_remove

        if (
            V.graph.wrapper_code.supports_intermediate_hooks  # type: ignore[has-type]
            and config.generate_intermediate_hooks
        ):
            # Not every node in the schedule will actually be live on output;
            # we can't check dead buffers.
            live_outs = kernels[0].args.live_output_buffers()
            for node in kernel_features.scheduler_nodes():
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

        self.free_buffers_in_scheduler()

    def create_kernel_choices(
        self, kernel_features: SIMDKernelFeatures, kernel_args, kernel_kwargs
    ) -> list[SIMDKernel]:
        return [
            self.kernel_type(
                *kernel_args,
                **kernel_kwargs,
            )
        ]

    def codegen_node_schedule_with_kernel(self, node_schedule, kernel):
        with kernel:
            stack = contextlib.ExitStack()
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
            for node in node_schedule:
                if node is DisableReduction:
                    stack.enter_context(kernel.disable_reduction())
                elif node is EnableReduction:
                    stack.close()
                else:
                    # TODO - use split ranges ?
                    indexing_dtype_strength_reduction(node._body)
                    index_vars = kernel.split_and_set_ranges(node.get_ranges())
                    node.codegen(index_vars)

    def _codegen_single_template(
        self,
        kernel,
        render,
        template_node,
        epilogue_nodes,
        prologue_nodes,
        *,
        only_gen_src_code=False,
    ):
        """
        Helper method to codegen a single template kernel variant
        """
        buf_name_to_prologue_group = {}
        template_reads = template_node.used_buffer_names()
        prologue_group = []
        for prologue in prologue_nodes:
            names = prologue.get_buffer_names()
            prologue_group.append(prologue)
            # this must be the end of a prologue group
            if names & template_reads:
                assert len(names) == 1
                buf_name_to_prologue_group[next(iter(names))] = prologue_group
                kernel.prologue_fused_inputs.add(next(iter(names)))
                prologue_group = []

        # all prologue groups should have finalized with use in template
        assert len(prologue_group) == 0

        # External template handlers (e.g. Helion) can override codegen
        result = kernel.codegen_template_override(
            self,
            template_node,
            epilogue_nodes,
            prologue_nodes,
            buf_name_to_prologue_group,
            prologue_preserves_zero_mask,
            render,
            only_gen_src_code,
        )
        if result is not None:
            return result

        with kernel:
            if not only_gen_src_code:
                # prologue nodes can only be fused if their only use is in the template,
                # so they are necessarily not allocated
                for node in [template_node, *epilogue_nodes]:
                    node.mark_run()

            partial_code = render()

            num_store_subgraphs = kernel.get_store_output_count()
            for i in range(num_store_subgraphs):
                subgraph_name = kernel._get_store_output_subgraph_name(i)
                with kernel.set_subgraph_body(subgraph_name):
                    for node in epilogue_nodes:
                        node.codegen(kernel.split_and_set_ranges(node.get_ranges()))
                    kernel.cse.invalidate(OrderedSet())

            for input_name, buffer in kernel.named_input_nodes.items():
                subgraph_name = f"<LOAD_INPUT_{input_name}>"
                if prologue_group := buf_name_to_prologue_group.get(
                    buffer.get_name(), []
                ):
                    can_codegen_without_upcast = all(
                        p_n.can_codegen_without_upcasts() for p_n in prologue_group
                    )

                    # TODO - this doesn't work with libdevice calls, potentially other bugs
                    # upcasting to fp32 and downcasting gives large slowdown
                    with config.patch(
                        "triton.codegen_upcast_to_fp32", not can_codegen_without_upcast
                    ):
                        with kernel.set_subgraph_body(subgraph_name):
                            for prologue_node in prologue_group:
                                if (
                                    len(prologue_node.get_buffer_names()) == 1
                                    and len(prologue_group) == 1
                                ):
                                    if prologue_preserves_zero_mask(prologue_node):
                                        kernel.prologue_fused_inputs_preserve_zero |= (
                                            prologue_node.get_buffer_names()
                                        )

                                prologue_node.codegen(
                                    kernel.split_and_set_ranges(
                                        prologue_node.get_ranges()
                                    )
                                )
                            kernel.cse.invalidate(OrderedSet())

        # Template hooks must be finalised after kernel.remove_kernel_local_buffers
        # is called (this is called when the kernel context is exited above), and when
        # the kernel handler is set (as below). This is because the hooks may add
        # DeferredLine type lines, which preclude lines involving buffers that have
        # been removed

        # finalize must be called after adding epilogue above
        with V.set_kernel_handler(kernel):
            if not isinstance(partial_code, str):
                # This is used to calculate flops in TritonTemplateKernels
                with ir.IRNode.current_origins(template_node.node.origins):
                    partial_code.finalize_hook("<DEF_KERNEL>")
                partial_code.finalize_hook("<ARGDEFS>", strict=False)

            # TODO: Maybe unify CUDATemplateKernel to also use PartialRender for flexible epilogue fusion.

            for input_name in kernel.named_input_nodes:
                subgraph_name = f"<LOAD_INPUT_{input_name}>"

                partial_code.finalize_hook(subgraph_name, strict=False)

            num_store_subgraphs = kernel.get_store_output_count()
            for i in range(num_store_subgraphs):
                subgraph_name = kernel._get_store_output_subgraph_name(i)

                partial_code.finalize_hook(subgraph_name)

            if isinstance(partial_code, str):
                src_code = partial_code
            else:
                # Ensure all hooks are finalized before the kernel is defined.
                # Note: some of these hooks may have been registered by a kernel subclass
                src_code = partial_code.finalize_remaining()

            node_schedule = [*prologue_nodes, template_node, *epilogue_nodes]

            if config.benchmark_kernel:
                num_gb = kernel.estimate_kernel_num_bytes() / 1e9
                src_code = (
                    f"{kernel.imports_for_benchmark_kernel()}\n"
                    f"{src_code}\n"
                    f"{kernel.codegen_kernel_benchmark(num_gb).getvalue()}"
                )

            if only_gen_src_code:
                return src_code

            kernel.kernel_name = self.define_kernel(src_code, node_schedule, kernel)

            return kernel

    def _get_multikernel_shapes(
        self, node: MultiTemplateBuffer
    ) -> tuple[tuple[int, ...], ...]:
        from ..ir import IRNode

        def get_size(arg):
            if not isinstance(arg, IRNode):
                return None
            if isinstance(arg, ir.BaseView):  # triton templates want the base tensor.
                arg = arg.unwrap_view()
            if (size := arg.maybe_get_size()) is None:
                return None
            return tuple(s for s in size)

        out = []
        for arg in list(node.inputs) + [node]:
            if isinstance(arg, (list, tuple)):
                out.append(tuple(get_size(_arg) for _arg in arg))
            else:
                out.append(get_size(arg))
        return tuple(out)

    def _kernel_has_dynamic_shapes(self, node: MultiTemplateBuffer) -> bool:
        shapes = self._get_multikernel_shapes(node)
        return any(
            any(
                isinstance(s, sympy.Expr) and not isinstance(s, sympy.Integer)
                for s in shape
            )
            for shape in shapes
        )

    def _make_shape_cache_key(
        self, node: MultiTemplateBuffer, hint: int
    ) -> tuple[tuple[int, ...], ...]:
        """
        Returns cache key for hint-based multi-graph; key is tuple of shapes with hint filled in.
        """
        shapes = self._get_multikernel_shapes(node)
        return tuple(
            tuple(
                hint
                if isinstance(s, sympy.Expr) and not isinstance(s, sympy.Integer)
                else s
                for s in shape
            )
            for shape in shapes
        )

    def codegen_template(
        self,
        template_node,
        epilogue_nodes,
        prologue_nodes,
        *,
        only_gen_src_code=False,
        hint_override: Optional[int] = None,
    ) -> Optional[str]:
        """
        Codegen a triton template with multi-kernel dispatch support

        If `only_gen_src_code=True` the src code will be returned instead of being
        codegenned into the wrapper
        """

        _, (_numel, rnumel) = template_node.group
        assert rnumel == 1

        if (
            isinstance(template_node.node, MultiTemplateBuffer)
            and template_node.node._make_kernel_renders
            and len(template_node.node._make_kernel_renders) > 1
            and self._kernel_has_dynamic_shapes(template_node.node)
        ):
            kernels = {}
            src_codes = []

            for (
                size_hint,
                make_kernel_render,
            ) in template_node.node._make_kernel_renders.items():
                kernel, render = make_kernel_render(
                    template_node.node, hint_override=hint_override
                )

                if only_gen_src_code:
                    src_code = self._codegen_single_template(
                        kernel,
                        render,
                        template_node,
                        epilogue_nodes,
                        prologue_nodes,
                        only_gen_src_code=True,
                    )
                    assert isinstance(src_code, str)
                    # pyrefly: ignore [bad-argument-type]
                    src_codes.append(src_code)
                else:
                    if size_hint is None:
                        continue  # skip kernel generation based on real runtime value; only use hints
                    kernel = self._codegen_single_template(
                        kernel,
                        render,
                        template_node,
                        epilogue_nodes,
                        prologue_nodes,
                        only_gen_src_code=False,
                    )
                    shape_cache_key = (
                        None
                        if size_hint is None
                        else self._make_shape_cache_key(template_node.node, size_hint)
                    )
                    kernels[shape_cache_key] = kernel

            if only_gen_src_code:
                return "\n\n".join(src_codes)

            MultiKernel.merge_workspaces_inplace(list(kernels.values()))
            multi_kernel = SizeHintMultiKernel(kernels)
            node_schedule = [*prologue_nodes, template_node, *epilogue_nodes]
            self.codegen_comment(node_schedule, multi_kernel.kernel_name)
            multi_kernel.call_kernel(multi_kernel.kernel_name)
            V.graph.removed_buffers |= multi_kernel.removed_buffers
            V.graph.inplaced_to_remove |= multi_kernel.inplaced_to_remove
            self.free_buffers_in_scheduler()
            return None
        else:
            kernel, render = template_node.node.make_kernel_render(
                template_node.node, hint_override=hint_override
            )

            if only_gen_src_code:
                return self._codegen_single_template(
                    kernel,
                    render,
                    template_node,
                    epilogue_nodes,
                    prologue_nodes,
                    only_gen_src_code=True,
                )
            else:
                kernel = self._codegen_single_template(
                    kernel,
                    render,
                    template_node,
                    epilogue_nodes,
                    prologue_nodes,
                    only_gen_src_code=False,
                )

                node_schedule = [*prologue_nodes, template_node, *epilogue_nodes]
                self.codegen_comment(node_schedule, kernel.kernel_name)
                kernel.call_kernel(kernel.kernel_name, template_node.node)

                V.graph.removed_buffers |= kernel.removed_buffers
                V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
                self.free_buffers_in_scheduler()
                return None

    def codegen_sync(self):
        V.graph.wrapper_code.writeline(V.graph.device_ops.synchronize())

    def generate_combo_kernel_code(
        self,
        subkernel_nodes: list[BaseSchedulerNode],
        custom_part_algorithm: bool,
        enable_autotune: bool,
        mixed_sizes: bool,
        only_gen_src_code: bool = False,
    ) -> list[tuple[str, Any, Any]]:
        from .triton_combo_kernel import ComboKernel

        fused_node_lists = [node.get_nodes() for node in subkernel_nodes]
        subkernel_map, node_schedule_map = {}, {}
        for pn, nodes in zip(subkernel_nodes, fused_node_lists):
            _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group
            node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
            tiling = self.select_tiling(node_schedule, numel, rnumel)
            node_schedule_map[pn] = node_schedule, tiling, numel, rnumel
            subkernel_map[pn] = ComboKernel.create_triton_kernel(
                tiling,
                features=SIMDKernelFeatures(node_schedule, numel, rnumel),
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
            if len(node_group) == 0:
                continue
            kernel = ComboKernel(
                enable_autotune=enable_autotune,
                mixed_sizes=mixed_sizes,
            )

            for pn in node_group:
                self.codegen_node_schedule_with_kernel(
                    node_schedule_map[pn][0],
                    kernel.create_sub_kernel(subkernel_map[pn]),
                )
                subkernel = subkernel_map[pn]
                node_schedule = node_schedule_map[pn][0]
                if not only_gen_src_code:
                    with V.set_kernel_handler(subkernel):  # type: ignore[call-arg]
                        for node in NodeScheduleMarker.only_nodes(node_schedule):
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
            self.codegen_comment(combo_kernel_node.snodes, kernel_name)
            log.debug("ComboKernels: generated kernel %s.", kernel_name)
            kernel.call_kernel(V.graph.wrapper_code, kernel_name)

        self.free_buffers_in_scheduler()

    @classmethod
    @functools.lru_cache(32)
    def candidate_tilings(cls, node, numel, reduction_numel) -> list[CandidateTiling]:
        is_pointwise = reduction_numel == 1

        def tile_ranges(is_pointwise: bool, ranges, rw) -> list[CandidateTiling]:
            """
            Compute tiling candidates by dividing up the iteration ranges.
            """
            assert len(rw.range_vars) == len(ranges), f"{rw.range_vars=} {ranges=}"

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
                if dep.name not in V.graph.removed_buffers
                and isinstance(dep, MemoryDep)
            ]
            write_names = OrderedSet([dep.name for dep in rw.writes])

            def collapse_ranges(ranges: Sequence[sympy.Expr]) -> sympy.Expr:
                return V.graph.sizevars.simplify(sympy_product(ranges))

            # Default to no tiling.
            tilings = [
                CandidateTiling(
                    tiling=cls.create_partial_tiling(
                        [collapse_ranges(ranges)], is_pointwise
                    ),
                    name="none",
                    score=0,
                )
            ]

            # Find non-trivial tiling candidates.
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
                    collapse_ranges(ranges[:split]),
                    collapse_ranges(ranges[split:]),
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
                    tilings.append(
                        CandidateTiling(
                            tiling=cls.create_partial_tiling(
                                [
                                    collapse_ranges(ranges[:split]),
                                    collapse_ranges(ranges[split:]),
                                ],
                                reduction_numel,
                            ),
                            score=score,
                            name=dep.name,
                        )
                    )

            return tilings

        pointwise_ranges, reduction_ranges = node.get_ranges()
        if (
            len(pointwise_ranges) <= 1
            and len(reduction_ranges) <= 1
            or free_unbacked_symbols(pointwise_ranges + reduction_ranges)
        ):
            return []

        # Tile either pointwise or reduction dims.
        pointwise_ranges, reduction_ranges = node.get_ranges()
        partial_tilings = tile_ranges(
            is_pointwise,
            pointwise_ranges if is_pointwise else reduction_ranges,
            node.pointwise_or_reduction_read_writes(is_pointwise),
        )

        # Fill in the missing ranges.
        full_tilings = [
            CandidateTiling(
                tiling=cls.complete_partial_tiling(
                    tiling.tiling, numel, reduction_numel
                ),
                score=tiling.score,
                name=tiling.name,
            )
            for tiling in partial_tilings
        ]

        return full_tilings

    @classmethod
    def create_tiling(
        cls, pw_tiling: Sequence[sympy.Expr], reduction_tiling: Sequence[sympy.Expr]
    ) -> immutable_dict[str, sympy.Expr]:
        """
        Create a tiling dict from pointwise and reduction splits.
        """
        pw_prefixes = ["z", "y", "x"][-len(pw_tiling) :]
        reduction_prefixes = ["r0_", "r1_"][: len(reduction_tiling)]
        return immutable_dict(
            [*zip(pw_prefixes, pw_tiling), *zip(reduction_prefixes, reduction_tiling)]
        )

    @classmethod
    def create_partial_tiling(
        cls,
        tiling: Sequence[sympy.Expr],
        is_pointwise: bool,
    ) -> immutable_dict[str, sympy.Expr]:
        return cls.create_tiling(
            tiling if is_pointwise else [],
            tiling if not is_pointwise else [],
        )

    @classmethod
    def complete_partial_tiling(
        cls,
        tiling: dict[str, sympy.Expr],
        numel: sympy.Expr,
        reduction_numel: sympy.Expr,
    ) -> immutable_dict[str, sympy.Expr]:
        """
        Given a tiling for only pointwise or reduction dimensions, adds the missing one.
        """
        splits = list(tiling.values())
        is_pointwise = "x" in tiling

        total_numel = numel * reduction_numel
        missing_tiling = [total_numel / sympy_product(splits)]

        tiling_args = (
            (splits, missing_tiling) if is_pointwise else (missing_tiling, splits)
        )
        return cls.create_tiling(*tiling_args)

    @classmethod
    def get_nd_tilings(
        cls,
        node_schedule,
        pointwise_numel,
        reduction_numel,
    ) -> list[immutable_dict[str, sympy.Expr]]:
        """
        Creates N-dimensional tiling candidates, attempting to simplify loads/stores
        by tiling the kernel into higher dimensions.

        Returns a list of tilings ranked by dimensionality.
        """
        is_pointwise = reduction_numel == 1
        tilings = OrderedSet[immutable_dict[str, sympy.Expr]]()
        for node in EnableReduction.filter(node_schedule):
            if not isinstance(node, scheduler.SchedulerNode):
                continue

            # If this is a reduction schedule, skip nodes which are missing their
            # reduction ranges.
            node_ranges = node.get_ranges()
            if not is_pointwise and len(node_ranges[1]) == 0:
                continue

            # Use the node ranges as the default tiling candidate.
            ranges_to_tile = node_ranges[0 if is_pointwise else 1]
            node_tilings = [ranges_to_tile]

            # Search the indexing expressions for more candidates.
            # If we see modular indexing, try to subdivide ranges into their implied
            # block shape.
            memory_deps = [
                dep
                for dep in node.read_writes.reads_and_writes()
                if isinstance(dep, MemoryDep) and len(dep.ranges) > 0
            ]
            for dep in memory_deps:
                # Attempt to partition variable ranges into pointwise and reduction groups.
                # To achieve this, merge the leading ranges until we reach the pointwise numel.
                all_var_ranges = [*dep.ranges.items()]
                pointwise_vars_numel = sympy.S.One
                sizevars = V.graph.sizevars
                pointwise_end_idx = 0
                for idx, (_var, numel) in enumerate(all_var_ranges):
                    pointwise_vars_numel *= numel
                    pointwise_end_idx = idx
                    if sizevars.statically_known_geq(
                        pointwise_vars_numel, pointwise_numel
                    ):
                        break

                # Reject the split if it does not match the total pointwise numel.
                if not sizevars.statically_known_equals(
                    pointwise_vars_numel, pointwise_numel
                ):
                    continue

                # Partition var ranges into pointwise and reduction splits.
                reduction_start_idx = pointwise_end_idx + 1
                var_ranges = (
                    all_var_ranges[:reduction_start_idx]
                    if is_pointwise
                    else all_var_ranges[reduction_start_idx:]
                )

                # Pattern match the subexpression pertaining to each index variable.
                index_tiling = []
                for var, numel in var_ranges:
                    index = BlockPatternMatcher.get_subexpr_involving_symbol(
                        dep.index, var
                    )

                    # Heuristic to bound the maximum dimensionality of the block.
                    num_dims = max(
                        2,
                        index.count(FloorDiv) + index.count(ModularIndexing),
                        len(ranges_to_tile),
                    )

                    # Attempt to pattern match the index expr.
                    # Failed matches default to the full range.
                    match_result = BlockPatternMatcher.match_mod_div_block_expr(
                        index, var, numel, num_dims
                    )
                    dims = match_result[0] if match_result is not None else [numel]
                    index_tiling.extend(dims)

                # Prune dimensions of size 1.
                index_tiling = [
                    dim
                    for dim in index_tiling
                    if not V.graph.sizevars.statically_known_equals(dim, sympy.S.One)
                ]

                if len(index_tiling) > 0:
                    node_tilings.append(index_tiling)

            # Flatten leading dimensions, assigning labels to each dim.
            for node_tiling in node_tilings:
                num_leading_dims = max(0, len(node_tiling) - get_max_tiles(2))
                first_trailing_dim = num_leading_dims + 1
                collapsed_leading_dim = sympy_product(node_tiling[:first_trailing_dim])
                collapsed_splits = (collapsed_leading_dim,) + tuple(
                    node_tiling[first_trailing_dim:]
                )
                tilings.add(
                    cls.complete_partial_tiling(
                        cls.create_partial_tiling(collapsed_splits, is_pointwise),
                        pointwise_numel,
                        reduction_numel,
                    )
                )

        # Rank tilings by the number of dimensions. E.g., prefer 2D to 1D.
        # Since this is a stable sort, ties are broken by schedule order.
        ranked_tilings = sorted(
            tilings,
            key=len,
            reverse=True,
        )

        return ranked_tilings

    @classmethod
    def compute_tiling_strategy(
        cls,
        node_schedule: list[NodeScheduleEntry],
        pointwise_numel: sympy.Expr,
        reduction_numel: sympy.Expr,
        coalesce_analysis: CoalesceVarAnalysis,
    ) -> tuple[dict[str, sympy.Expr], Optional[dict[str, sympy.Expr]]]:
        """
        Generates a tiling, and a score of each tile according to each tile's coalesced memory accesses.
        """
        tiling_var: Optional[sympy.Expr] = (
            None
            if not coalesce_analysis.suggested_split
            else coalesce_analysis.suggested_split.var
        )

        all_iter_vars = coalesce_analysis.norm_read_writes.index_vars
        all_red_vars = coalesce_analysis.norm_read_writes.reduce_vars
        ranges = coalesce_analysis.norm_read_writes.var_ranges

        pw_ranges = [ranges[v] for v in all_iter_vars]
        red_ranges = [ranges[v] for v in all_red_vars]

        # Sometimes dynamic shapes is unable to prove equality without hint
        get_hint = functools.partial(
            V.graph.sizevars.size_hint, fallback=config.unbacked_symint_fallback
        )
        torch._check(
            get_hint(sympy_product(pw_ranges)) == get_hint(pointwise_numel),
            lambda: f"{pw_ranges}, {pointwise_numel}, {node_schedule}",
        )

        torch._check(
            get_hint(sympy_product(red_ranges)) == get_hint(reduction_numel),
            lambda: f"{red_ranges}, {reduction_numel}, {node_schedule}",
        )

        # score of a pointwise or reduction split
        scored_sub_split: dict[Any, tuple[list[int], list[int]]] = {}

        score_split: list[
            tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]
        ] = []

        def process_node_vars(
            vars_to_use: tuple[sympy.Expr, ...] = (),
            use_split_var: bool = False,
            is_pointwise: bool = False,
        ) -> tuple[list[int], list[int]]:
            """
            Generate a tiling, and a tiling score, given vars to use as splits.
            """

            ranges = pw_ranges if is_pointwise else red_ranges
            target_numel = pointwise_numel if is_pointwise else reduction_numel
            # Some kernels have no reduction ranges, and a reduction numel of 1
            if not ranges:
                if target_numel:
                    return ([target_numel], [])
                else:
                    return ([], [])

            key = (repr(vars_to_use), use_split_var, is_pointwise)
            if out := scored_sub_split.get(key):
                return out

            splitting_vars = all_iter_vars if is_pointwise else all_red_vars

            splits = []
            split_scores = []
            prod = 1
            prev_var_coalesced_score = 0

            # iterate from non-dense to dense
            for v, v_range in zip(splitting_vars, ranges):
                if v not in vars_to_use:
                    prod *= v_range
                    prev_var_coalesced_score = coalesce_analysis.coalesced_by_var.get(
                        v, 0
                    )
                    continue

                if use_split_var and v == tiling_var:
                    var_tiling = coalesce_analysis.suggested_split
                    assert var_tiling is not None

                    tile = var_tiling.tiling_factor
                    remainder = FloorDiv(v_range, var_tiling.tiling_factor)

                    splits.append(prod * remainder)
                    split_scores.append(var_tiling.score)

                    splits.append(tile)
                    split_scores.append(coalesce_analysis.coalesced_by_var.get(v, 0))

                    prod = 1
                    prev_var_coalesced_score = 0

                    continue

                prod *= v_range
                splits.append(prod)
                split_scores.append(coalesce_analysis.coalesced_by_var.get(v, 0))
                prod = 1

            if prod != 1 or (is_pointwise and len(splits) == 0):
                splits.append(prod)
                split_scores.append(prev_var_coalesced_score)

            # penalize splits that leave small blocks
            # where we can't fully utilize full memory transaction
            # TODO: incorporate exact bitwidth, and read/write
            # coalesced write is 2x more important
            for i in range(len(splits)):
                s = V.graph.sizevars.size_hint(splits[i], fallback=32)
                s = min(s, 8)
                split_scores[i] = int(split_scores[i] * s / 8)

            scored_sub_split[key] = (splits, split_scores)
            return (splits, split_scores)

        # add the default tiling
        score_split.append(
            (
                process_node_vars(is_pointwise=True),
                process_node_vars(is_pointwise=False),
            )
        )

        if tiling_var:
            score_split.append(
                (
                    process_node_vars(
                        (tiling_var,), use_split_var=True, is_pointwise=True
                    ),
                    process_node_vars(is_pointwise=False),
                )
            )

        # TODO, add tests, reduction splits if config.triton.tile_reductions
        # TODO: we should ignore tiny increases in score for extra splits
        overlapping_iter_vars = (
            all_iter_vars & coalesce_analysis.coalesced_by_var.keys()
        )
        for v in overlapping_iter_vars:
            score_split.append(
                (
                    process_node_vars((v,), is_pointwise=True),
                    process_node_vars(is_pointwise=False),
                )
            )

        if get_max_tiles(default=3) == 3 and reduction_numel == 1:
            for vars_to_use in itertools.combinations(overlapping_iter_vars, 2):
                score_split.append(
                    (
                        process_node_vars(vars_to_use, is_pointwise=True),
                        process_node_vars(is_pointwise=False),
                    )
                )

        tilings: list[tuple[CandidateTiling, immutable_dict[str, sympy.Expr]]] = []
        for (pw_split, pw_score), (red_split, red_score) in score_split:
            candidate = CandidateTiling(
                cls.create_tiling(pw_split, red_split),
                score=sum(pw_score) + sum(red_score),
            )
            tiling_score = cls.create_tiling(pw_score, red_score)
            tilings.append((candidate, tiling_score))

        default_tiling = cls.create_tiling([pointwise_numel], [reduction_numel])

        # add a slight penalty for longer tilings that dont increase score much,
        # and are poor sizes
        bad_size_additional_tiling_penalty = 1.025
        good_size_tiling_penalty = 1.005

        total_uncoalesced = sum(coalesce_analysis.uncoalesced_addrs.values())

        def score_mod(t):
            score_factor = 1.0
            for tile_size in t[0].tiling.values():
                if not CandidateTiling.is_good_size(tile_size):
                    score_factor = score_factor / bad_size_additional_tiling_penalty
                else:
                    score_factor = score_factor / good_size_tiling_penalty

            # Add uncoalesced memory score to prevent small coalesced benefits
            # from dominating large amounts of uncoalesced memory
            uncoalesced_penalty = total_uncoalesced * 0.05

            return -(t[0].score + uncoalesced_penalty) * score_factor

        # apply penalty for longer tilings that dont increase score much
        for cand, tiling_score in sorted(tilings, key=score_mod):
            if (
                cls.tiling_is_compatible(
                    node_schedule, pointwise_numel, reduction_numel, cand.tiling
                )
                or cand.tiling == default_tiling
            ):
                # we always include default reduction numel == 1, dont include
                tiling_len = len(cand.tiling) - (1 if reduction_numel == 1 else 0)
                if tiling_len > get_max_tiles(default=3):
                    perf_hint_log.info(
                        "Found optimal tiling with %s tiles but torch._inductor.config.triton.max_tiles "
                        "set to %s. Consider increasing",
                        tiling_len,
                        torch._inductor.config.triton.max_tiles,
                    )
                    continue

                return cand.tiling, tiling_score

            # surprisingly, the default tiling is not always read as compatible by `tiling_is_compatible`
            # TODO - look into, occurs with dynamic shapes often
            if cand.tiling == default_tiling:
                return cand.tiling, tiling_score

        return default_tiling, None

    @classmethod
    def tiling_is_compatible(
        cls,
        node_schedule: list[NodeScheduleEntry],
        numel: sympy.Expr,
        reduction_numel: sympy.Expr,
        tiling: dict[str, sympy.Expr],
    ):
        assert isinstance(tiling, dict)
        return all(
            SIMDKernel.is_compatible(
                tiling.values(), node.get_ranges(), reduction_numel=reduction_numel
            )
            for node in node_schedule
            if isinstance(node, scheduler.SchedulerNode)
        )

    @classmethod
    def get_first_compatible_tiling(
        cls,
        node_schedule: list[NodeScheduleEntry],
        numel: sympy.Expr,
        reduction_numel: sympy.Expr,
        ranked_tilings: list[dict[str, sympy.Expr]],
    ):
        for tiling in ranked_tilings:
            if cls.tiling_is_compatible(node_schedule, numel, reduction_numel, tiling):
                return tiling

        return None

    @classmethod
    def select_tiling(
        cls,
        node_schedule,
        numel,
        reduction_numel=sympy.S.One,
        coalesce_analysis: Optional[CoalesceVarAnalysis] = None,
    ) -> dict[str, sympy.Expr]:
        return cls.get_tiling_and_scores(
            node_schedule, numel, reduction_numel, coalesce_analysis
        )[0]

    @classmethod
    def get_tiling_and_scores(
        cls,
        node_schedule,
        numel,
        reduction_numel=sympy.S.One,
        coalesce_analysis: Optional[CoalesceVarAnalysis] = None,
    ) -> tuple[dict[str, sympy.Expr], Optional[dict[str, sympy.Expr]]]:
        """
        Heuristics to decide how to tile kernels.
        Currently, we tile based on stride-1 dimensions.

        Returns:
            `(tile1, tile2, reduction_numel)` s.t. `tile1 * tile2 == numel`

        """
        # If this is a reduction, only tile reduction dims.
        is_pointwise = reduction_numel == 1

        # Tiled reductions are gated by a config flag.
        default_tiling = cls.create_tiling([numel], [reduction_numel])

        # Force tiling compatible with matmul dimensions
        # when natively generating matmul without template calls.
        for node in EnableReduction.filter(node_schedule):
            if isinstance(node.node, ir.ComputedBuffer):
                if (
                    node.node.get_reduction_type() == "dot"
                    and config.triton.native_matmul
                ):
                    # A[M,K] @ B[K,N]
                    # force tiling to be {'y':M, 'x':N, 'r0_':K}
                    node_ranges = node.get_ranges()
                    range_y_x = node_ranges[0]  # (M,N)
                    range_r = node_ranges[1]  # (K)
                    tiling = cls.create_tiling(range_y_x, range_r)
                    return tiling, None

        # # TODO: enable by default
        if (
            torch._inductor.config.triton.coalesce_tiling_analysis
            and coalesce_analysis
            and not config.triton.prefer_nd_tiling
        ):
            return cls.compute_tiling_strategy(
                node_schedule, numel, reduction_numel, coalesce_analysis
            )

        if (not is_pointwise and not config.triton.tile_reductions) or get_max_tiles(
            default=2
        ) <= 1:
            # Emit a perf hint in case we miss an opportunity to tile a reduction.
            if perf_hint_log.level <= logging.WARNING:
                for node in EnableReduction.filter(node_schedule):
                    if (
                        not config.triton.tile_reductions
                        and len(cls.candidate_tilings(node, numel, reduction_numel)) > 0
                    ):
                        perf_hint_log.info(
                            textwrap.dedent(
                                """
                                Reduction over non-contiguous dims.
                                Consider setting config.triton.tile_reductions to True.
                                """
                            )
                        )
                        break

            return default_tiling, None

        seen_names: OrderedSet[str] = OrderedSet()
        candidate_tiles: Counter[CandidateTiling] = collections.Counter()
        for node in EnableReduction.filter(node_schedule):
            for candidate_tiling in cls.candidate_tilings(node, numel, reduction_numel):
                if candidate_tiling.name in seen_names:
                    continue
                elif candidate_tiling.name is not None:
                    seen_names.add(candidate_tiling.name)
                candidate_tiles[candidate_tiling] += candidate_tiling.score

        ranked_tilings: list[dict[str, sympy.Expr]] = [
            candidate_tiling.tiling
            for candidate_tiling, score in candidate_tiles.most_common()
        ]

        if get_max_tiles(default=2) >= 3 and is_pointwise:
            # Consider adding a third dimension of tiling, but only
            # when a1 is a multiple of b1; otherwise, you have a lot
            # of stragglers which is annoying to generate code for.
            #
            # NB: More than three max tiles is not enabled by default.

            def convert_tiling_to_3d(
                tiling0: dict[str, sympy.Expr], tiling1: dict[str, sympy.Expr]
            ) -> Optional[dict[str, sympy.Expr]]:
                a0, a1 = tiling0["x"], tiling0.get("y", 1)
                b0, b1 = tiling1["x"], tiling1.get("y", 1)

                if (
                    free_unbacked_symbols([a1, b1])
                    or V.graph.sizevars.size_hint(a1 - b1) == 0
                ):
                    return None
                if V.graph.sizevars.size_hint(a1 - b1) < 0:
                    # swap so a0 is bigger
                    (a0, a1), (b0, b1) = (b0, b1), (a0, a1)

                assert V.graph.sizevars.size_hint(a1 - b1) > 0
                if not V.graph.sizevars.statically_known_multiple_of(a1, b1):
                    return None

                new_tiling = {
                    "z": a0,
                    "y": FloorDiv(a1, b1),
                    "x": b1,
                    "r0_": tiling0["r0_"],
                }

                return new_tiling

            for i in range(1, len(ranked_tilings)):
                new_3d_tiling = convert_tiling_to_3d(
                    ranked_tilings[0], ranked_tilings[i]
                )
                if new_3d_tiling is not None:
                    ranked_tilings = [new_3d_tiling] + ranked_tilings
                    break  # only 1 choice for now

        if len(ranked_tilings) > 1:
            perf_hint_log.info("possibly bad tiling: %s", ranked_tilings)

        # Optionally, prefer tiling into as many dimensions as possible.
        # pyrefly: ignore [unbound-name]
        if config.triton.prefer_nd_tiling:
            ranked_tilings = (
                cls.get_nd_tilings(node_schedule, numel, reduction_numel)
                + ranked_tilings
            )

        if tiling := cls.get_first_compatible_tiling(
            node_schedule, numel, reduction_numel, ranked_tilings
        ):
            return tiling, None

        return default_tiling, None

    def flush(self):
        pass

    def ready_to_flush(self) -> bool:
        return False

    def generate_kernel_code_from_nodes(
        self, nodes, benchmark_kernel=False, hint_override: Optional[int] = None
    ):
        if not any(n.is_template() for n in nodes):
            _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group
            node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
            tiling = self.select_tiling(node_schedule, numel, rnumel)
            kernel = self.kernel_type(
                tiling,
                features=SIMDKernelFeatures(node_schedule, numel, rnumel),
            )
            self.codegen_node_schedule_with_kernel(node_schedule, kernel)
            with (
                config.patch("benchmark_kernel", benchmark_kernel),
                V.set_kernel_handler(kernel),
            ):
                src_code = kernel.codegen_kernel()
        else:
            prologue, template, epilogue = nodes[0].get_prologue_template_epilogue(
                nodes
            )
            with config.patch("benchmark_kernel", benchmark_kernel):
                src_code = self.codegen_template(
                    template,
                    epilogue,
                    prologue,
                    only_gen_src_code=True,
                    hint_override=hint_override,
                )

        # pyrefly: ignore [missing-attribute]
        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), "triton_")
        return src_code

    def define_kernel(self, src_code, node_schedule, kernel):
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class CandidateTiling:
    tiling: dict[str, sympy.Expr]
    score: int  # higher is better
    name: Optional[str] = None

    @staticmethod
    def is_good_size(s):
        """Somewhat arbitrary heuristic used to boost scores for some sizes"""
        s = V.graph.sizevars.size_hint(s, fallback=8192)
        return s >= 32 and (s % 32 == 0)


class CantSplit(Exception):
    def __init__(self, expr, remaining):
        super().__init__()
        self.expr = expr
        self.remaining = remaining

    def __str__(self):
        return f"{self.expr} not divisible by {self.remaining}"
