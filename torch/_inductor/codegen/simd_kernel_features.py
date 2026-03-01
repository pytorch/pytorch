from __future__ import annotations

import collections
import dataclasses
import functools
import itertools
import typing
from typing import Any

import sympy

import torch

from ...utils._ordered_set import OrderedSet
from ...utils._sympy.functions import FloorDiv, ModularIndexing
from ...utils._sympy.symbol import make_symbol, SymT
from ..dependencies import Dep, extract_loop_body_with_args, MemoryDep
from ..runtime.hints import ReductionHint
from ..scheduler import SchedulerNode
from ..utils import cache_on_self
from ..virtualized import V


if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from torch._inductor.tiling_utils import CoalesceVarAnalysis


class NodeScheduleMarker:
    @staticmethod
    def only_nodes(it: Iterable[NodeScheduleEntry]) -> Iterable[SchedulerNode]:
        for item in it:
            if not (item is DisableReduction or item is EnableReduction):
                yield item  # type: ignore[misc]

    @staticmethod
    def is_reduction() -> bool:
        return False


NodeScheduleEntry = SchedulerNode | type[NodeScheduleMarker]


class DisableReduction(NodeScheduleMarker):
    """
    Marker to invoke `kernel.disable_reduction()`.  This closes a
    reduction loop and allows for pointwise ops to occur on the output
    of a reduction.
    """


class EnableReduction(NodeScheduleMarker):
    """
    Marker to end a DisableReduction block.
    """

    @staticmethod
    def filter(node_schedule: list[NodeScheduleEntry]) -> Iterable[SchedulerNode]:
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
                yield node  # type: ignore[misc]


class SIMDKernelFeatures:
    """
    An ordered schedule of nodes that will become a single kernel.
    """

    def __init__(
        self,
        node_schedule: list[NodeScheduleEntry],
        numel: sympy.Expr,
        reduction_numel: sympy.Expr = sympy.S.One,
        coalesce_analysis: CoalesceVarAnalysis | None = None,
    ):
        self.node_schedule = node_schedule
        # numel excludes reduction_numel
        self.numel: sympy.Expr = V.graph.sizevars.simplify(numel)
        self.reduction_numel: sympy.Expr = V.graph.sizevars.simplify(reduction_numel)
        self._stats_cache: dict[tuple[sympy.Expr, ...], MemoryStats] = {}
        self.coalesce_analysis = coalesce_analysis

    @cache_on_self
    def is_reduction(self) -> bool:
        return self.reduction_numel != 1

    @cache_on_self
    def scheduler_nodes(self) -> Iterable[SchedulerNode]:
        return tuple(NodeScheduleMarker.only_nodes(self.node_schedule))

    def reduction_nodes(self) -> list[SchedulerNode]:
        return [n for n in self.scheduler_nodes() if n.is_reduction()]

    @cache_on_self
    def buf_accesses(self) -> dict[str, list[Dep]]:
        """only needed for config.benchmark_kernel"""
        buf_accesses = collections.defaultdict(list)
        for node in self.scheduler_nodes():
            for access in node.read_writes.reads | node.read_writes.writes:
                buf_accesses[access.name].append(access)
        return buf_accesses

    @cache_on_self
    def op_counts(self) -> collections.Counter[str]:
        counts: collections.Counter[str] = collections.Counter()
        for node in self.scheduler_nodes():
            counts.update(node._body.op_counts)
        return counts

    def contains_op(self, op_name: str) -> bool:
        """True if V.ops.{op_name} is used in node_schedule"""
        return bool(self.op_counts().get(op_name))

    def get_mutations(self) -> OrderedSet[str]:
        mutations: OrderedSet[str] = OrderedSet()
        for node in self.scheduler_nodes():
            for buf in node.get_outputs():
                mutations.update(buf.get_mutations())
        return mutations

    @cache_on_self
    def select_index_dtype(self) -> torch.dtype:
        # Gather all used buffer names
        buffer_names: OrderedSet[str] = OrderedSet()
        for node in self.scheduler_nodes():
            buffer_names.update(node.get_buffer_names())
            buffer_names.update(node.used_buffer_names())
        buffers = [V.graph.get_buffer(name) for name in buffer_names]

        # In theory we can separately check xnumel and rnumel are <= int_max
        # but some indexers do use the full linear index so we need to be
        # conservative here.
        total_numel = self.numel * self.reduction_numel

        from .simd import SIMDScheduling

        if SIMDScheduling.can_use_32bit_indexing(total_numel, buffers):
            return torch.int32
        return torch.int64

    def get_reduction_hint(
        self, tiling_scores: dict[str, int] | None = None
    ) -> ReductionHint:
        reductions = self.reduction_nodes()
        if len(reductions) > 0:
            hints = [self.reduction_hint(n) for n in reductions]
            if hints.count(hints[0]) == len(hints):
                reduction_hint_val = hints[0]
            else:
                reduction_hint_val = ReductionHint.DEFAULT

            if (
                reduction_hint_val == ReductionHint.INNER
                and self.has_non_contiguous_pw_in_reduction_kernel()
            ):
                reduction_hint_val = ReductionHint.DEFAULT

            # Upgrade DEFAULT to INNER for inner reductions based on tiling scores
            if (
                reduction_hint_val == ReductionHint.DEFAULT
                and tiling_scores is not None
                and "x" in tiling_scores
                and "r0_" in tiling_scores
            ):
                # If reduction dimension has much better coalescing than non-reduction dimensions,
                # this is an inner reduction
                from ..codegen.triton import INNER_REDUCTION_RATIO_THRESHOLD

                r_coalesce_ratio = tiling_scores["r0_"] / max(tiling_scores["x"], 1)
                contiguous_red = r_coalesce_ratio >= INNER_REDUCTION_RATIO_THRESHOLD
                if contiguous_red:
                    reduction_hint_val = ReductionHint.INNER
        else:
            reduction_hint_val = ReductionHint.DEFAULT
        return reduction_hint_val

    @cache_on_self
    def buffer_read_counts(self) -> dict[str, int]:
        """Counts how many times each buffer is read within the kernel"""
        read_counts: dict[str, int] = collections.defaultdict(int)

        for node in self.scheduler_nodes():
            # node.read_writes.reads contains MemoryDep objects for each read
            for read_dep in node.read_writes.reads:
                read_counts[read_dep.name] += 1

        return dict(read_counts)  # Convert defaultdict to regular dict

    def has_non_contiguous_pw_in_reduction_kernel(self) -> bool:
        pointwise_nodes = [
            n
            for n in self.scheduler_nodes()
            if not n.is_reduction()
            and n.group[1][0] == self.numel * self.reduction_numel
        ]
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

    @staticmethod
    def reduction_hint(node: Any) -> ReductionHint:
        assert node.is_reduction()
        if node.node.data.reduction_hint != ReductionHint.INNER and all(
            dep.is_contiguous()
            for dep in itertools.chain(node.read_writes.reads, node.read_writes.writes)
        ):
            return ReductionHint.INNER
        else:
            return node.node.data.reduction_hint

    def memory_stats(
        self, groups_dict: dict[str, sympy.Expr] | None = None
    ) -> MemoryStats:
        """Analysis to generate features that can be used in heuristics"""
        if groups_dict is None:
            groups = (self.numel, self.reduction_numel)
        elif groups_dict.keys() == OrderedSet(["x", "r0_"]):
            groups = (groups_dict["x"], groups_dict["r0_"])
        else:
            raise NotImplementedError(f"groups_dict={groups_dict!r}")
        result = self._stats_cache.get(groups)
        if result is None:
            self._stats_cache[groups] = result = MemoryStats.compute(
                MemoryEstimator(self, groups)
            )
        return result


class MemoryEstimator:
    """
    Estimate various properties of the kernel for use in heuristics.
    We simulate the memory effects of CSE/buffer elimination in codegen.
    """

    kernel_sizes: tuple[sympy.Expr, ...]
    outside_loop: MemoryEstimate
    loops: list[MemoryEstimate]
    persistent: MemoryEstimate
    symbols: list[sympy.Symbol]

    def __init__(self, features: SIMDKernelFeatures, groups: Sequence[sympy.Expr]):
        self.features = features
        self.inside_reduction = features.is_reduction()
        self.store_buffer_names: OrderedSet[str] = OrderedSet()
        self.must_keep_buffers: OrderedSet[str] = OrderedSet()
        self.num_reductions_dims = 1
        self.groups = groups
        self.symbols = [make_symbol(SymT.INDEX, i) for i in range(len(groups))]
        # We are doing two estimates simultaneously:
        # 1) the first is a for a non-persistent (aka looped) reduction, using self.outside_loop/self.loops
        # we add an item to loops each corresponding to each reduction loop in the kernel
        # outside_loop is only used for broadcasting or point-wise ops that don't use the reduction dimension
        # 2) the second is for a persistent kernel, using self.persistent
        # persistent kernels don't have loops, so we only have one MemoryEstimate()
        # for point-wise ops the two estimates will be the same, they matter for reductions only
        self.outside_loop = MemoryEstimate()
        self.loops = [MemoryEstimate()]
        self.persistent = MemoryEstimate()
        self.simulate_codegen()
        self.remove_kernel_local()

    def simulate_codegen(self) -> None:
        from .simd import SIMDKernel

        kernel_size_outside_loop = (*self.groups[:-1], sympy.S.One)
        kernel_size_inside_loop = tuple(self.groups)
        self.kernel_sizes = kernel_size_inside_loop

        for node in self.features.node_schedule:
            if node is DisableReduction:
                self.inside_reduction = False
                self.kernel_sizes = kernel_size_outside_loop
                continue
            elif node is EnableReduction:
                self.inside_reduction = True
                self.kernel_sizes = kernel_size_inside_loop
                self.loops.append(MemoryEstimate())
                continue
            assert isinstance(node, SchedulerNode)
            rw = extract_loop_body_with_args(
                node._body,
                SIMDKernel.map_kernel_groups_to_node_sizes(
                    self.kernel_sizes, node.get_ranges(), self.set_ranges
                ),
                dict(zip(self.symbols, self.kernel_sizes)),
            )

            for dep in rw._reads:
                if not isinstance(dep, MemoryDep):
                    continue
                dep = dep.simplify_with_ranges()
                if not self.persistent.writes.get(dep.name):  # cache miss?
                    self.persistent.reads[dep.name].add(dep)
                # the cache behavior of looped kernels is more complex than the persistent case above
                # some operations are lifted outside the loop (if they don't use the reduction dimension)
                # other operations are inside the loop, and can only be reused within the same loop
                if not (
                    self.outside_loop.writes.get(dep.name)
                    or self.loops[-1].writes.get(dep.name)
                ):
                    self.scope(dep).reads[dep.name].add(dep)
                    if dep.name in self.store_buffer_names and self.loops[-1].reads.get(
                        dep.name
                    ):
                        self.must_keep_buffers.add(dep.name)

            for dep in rw._writes:
                if not isinstance(dep, MemoryDep):
                    continue
                dep = dep.simplify_with_ranges()
                self.store_buffer_names.add(dep.name)
                self.persistent.writes[dep.name].add(dep)
                self.scope(dep).writes[dep.name].add(dep)

    def remove_kernel_local(self) -> None:
        # Remove any kernel-local buffers
        fused_node_names = OrderedSet(
            [n.get_name() for n in self.features.scheduler_nodes()]
        )
        for name in self.store_buffer_names:
            if not self.persistent.reads.get(
                name
            ) and V.graph.scheduler.can_buffer_be_removed_through_fusion(
                name, fused_node_names
            ):
                self.persistent.remove(name)
                if name not in self.must_keep_buffers:
                    # we can also remove this from the looped kernel
                    self.outside_loop.remove(name)
                    for loop in self.loops:
                        loop.remove(name)

        if not self.loops[-1]:
            self.loops.pop()  # for pointwise ops

    def scope(self, dep: MemoryDep) -> MemoryEstimate:
        """Determine how a read/write should be categorized"""
        if self.inside_reduction and (
            self.has_reduction_var(dep.index) or dep.is_indirect()
        ):
            return self.loops[-1]
        return self.outside_loop

    def has_reduction_var(self, index: sympy.Expr) -> bool:
        for sym in self.symbols[-self.num_reductions_dims :]:
            if isinstance(sym, sympy.Symbol) and sym in index.free_symbols:
                return True
        return False

    def set_ranges(self, *lengths: list[list[sympy.Expr]]) -> list[list[sympy.Expr]]:
        assert len(self.kernel_sizes) == len(lengths)
        return [
            self.make_flat_range(sym, numel, length)
            for sym, numel, length in zip(self.symbols, self.kernel_sizes, lengths)
        ]

    @staticmethod
    def make_flat_range(
        sym: sympy.Symbol, numel: sympy.Expr, lengths: list[sympy.Expr]
    ) -> list[sympy.Expr]:
        if len(lengths) == 1 and numel == lengths[0]:
            return [sym]
        divisor = sympy.S.One
        itervars = []
        for length in reversed(lengths):
            if V.graph.sizevars.statically_known_equals(divisor * length, numel):
                expr = FloorDiv(sym, divisor)
            else:
                expr = ModularIndexing(sym, divisor, length)
            itervars.append(expr)
            divisor = divisor * length
        return [*reversed(itervars)]


@dataclasses.dataclass
class MemoryEstimate:
    """Tracks the memory usage of a single loop in the generated kernel"""

    reads: dict[str, OrderedSet[MemoryDep]] = dataclasses.field(
        default_factory=functools.partial(collections.defaultdict, OrderedSet)
    )
    writes: dict[str, OrderedSet[MemoryDep]] = dataclasses.field(
        default_factory=functools.partial(collections.defaultdict, OrderedSet)
    )

    def remove(self, name: str) -> None:
        self.reads.pop(name, None)
        self.writes.pop(name, None)

    def __bool__(self) -> bool:
        return bool(self.reads or self.writes)

    def __repr__(self) -> str:
        return f"""MemoryEstimate(
            reads={[*itertools.chain.from_iterable(self.reads.values())]!r},
            writes={[*itertools.chain.from_iterable(self.writes.values())]!r}
        )"""


@dataclasses.dataclass
class StatsForDim:
    """Memory usage stats for a block dimension in the generated kernel (different from user dimensions)"""

    # the number of load/store ops
    count_per_thread_contiguous: int = 0
    count_per_thread_broadcast: int = 0
    count_per_thread_non_contiguous: int = 0  # excludes broadcast

    # total bytes in each load/store op for a single element
    bytes_per_thread_contiguous: int = 0
    bytes_per_thread_broadcast: int = 0
    bytes_per_thread_non_contiguous: int = 0  # excludes broadcast

    # total bytes read by entire kernel
    bytes_contiguous_or_broadcast: sympy.Expr = sympy.S.Zero
    bytes_non_contiguous: sympy.Expr = sympy.S.Zero

    def __add__(self, other: typing.Self) -> StatsForDim:
        return StatsForDim(
            count_per_thread_contiguous=self.count_per_thread_contiguous
            + other.count_per_thread_contiguous,
            count_per_thread_broadcast=self.count_per_thread_broadcast
            + other.count_per_thread_broadcast,
            count_per_thread_non_contiguous=self.count_per_thread_non_contiguous
            + other.count_per_thread_non_contiguous,
            bytes_per_thread_contiguous=self.bytes_per_thread_contiguous
            + other.bytes_per_thread_contiguous,
            bytes_per_thread_broadcast=self.bytes_per_thread_broadcast
            + other.bytes_per_thread_broadcast,
            bytes_per_thread_non_contiguous=self.bytes_per_thread_non_contiguous
            + other.bytes_per_thread_non_contiguous,
            bytes_contiguous_or_broadcast=self.bytes_contiguous_or_broadcast
            + other.bytes_contiguous_or_broadcast,
            bytes_non_contiguous=self.bytes_non_contiguous + other.bytes_non_contiguous,
        )

    @property
    def count_per_thread(self) -> int:
        return (
            self.count_per_thread_contiguous
            + self.count_per_thread_broadcast
            + self.count_per_thread_non_contiguous
        )

    @property
    def bytes_per_thread(self) -> int:
        return (
            self.bytes_per_thread_contiguous
            + self.bytes_per_thread_broadcast
            + self.bytes_per_thread_non_contiguous
        )

    @property
    def bytes(self) -> sympy.Expr:
        return self.bytes_contiguous_or_broadcast + self.bytes_non_contiguous

    @property
    def contiguous_score(self) -> float:
        return 1.0 - self.count_per_thread_non_contiguous / max(
            self.count_per_thread, 1
        )


@dataclasses.dataclass
class StatsForLoop:
    """Memory usage stats for single loop in the generated kernel"""

    # load/store ops
    count_per_thread: int = 0
    bytes_per_thread: int = 0

    def __add__(self, other: typing.Self) -> StatsForLoop:
        return StatsForLoop(
            count_per_thread=self.count_per_thread + other.count_per_thread,
            bytes_per_thread=self.bytes_per_thread + other.bytes_per_thread,
        )


@dataclasses.dataclass
class StatsForReadsOrWrites:
    """Memory usage stats that are collected for reads/writes/both"""

    dim: list[StatsForDim]
    loop: list[StatsForLoop]
    # total bytes contiguous in any dimension
    bytes_contiguous_or_broadcast: sympy.Expr = sympy.S.Zero
    bytes_non_contiguous: sympy.Expr = sympy.S.Zero

    def __add__(self, other: typing.Self) -> StatsForReadsOrWrites:
        assert len(self.dim) == len(other.dim)
        assert len(self.loop) == len(other.loop)
        return StatsForReadsOrWrites(
            dim=[a + b for a, b in zip(self.dim, other.dim)],
            loop=[a + b for a, b in zip(self.loop, other.loop)],
            bytes_contiguous_or_broadcast=self.bytes_contiguous_or_broadcast
            + self.bytes_contiguous_or_broadcast,
            bytes_non_contiguous=self.bytes_non_contiguous + other.bytes_non_contiguous,
        )

    @property
    def count_per_thread(self) -> int:
        return self.dim[0].count_per_thread

    @property
    def bytes_per_thread(self) -> int:
        return self.dim[0].bytes_per_thread

    @property
    def bytes(self) -> sympy.Expr:
        return self.bytes_contiguous_or_broadcast + self.bytes_non_contiguous

    @classmethod
    def compute(
        cls,
        loop_deps: list[dict[str, OrderedSet[MemoryDep]]],
        index_symbols: list[sympy.Symbol],
    ) -> typing.Self:
        ndim = len(index_symbols)
        result = cls(dim := [StatsForDim() for _ in range(ndim)], [])
        for dep_group in loop_deps:
            result.loop.append(loop_stats := StatsForLoop())
            for name, deps in dep_group.items():
                assert deps
                contiguous_or_broadcast = [True] * ndim
                numel = sympy.S.Zero
                itemsize = V.graph.get_dtype(name).itemsize
                loop_stats.count_per_thread += len(deps)
                loop_stats.bytes_per_thread += itemsize * len(deps)
                for dep in deps:
                    strides: list[sympy.Expr] = V.graph.sizevars.stride_vars(
                        dep.index, index_symbols
                    )
                    for i in range(ndim):
                        if V.graph.sizevars.statically_known_equals(strides[i], 1):
                            dim[i].count_per_thread_contiguous += 1
                            dim[i].bytes_per_thread_contiguous += itemsize
                        elif (
                            V.graph.sizevars.statically_known_equals(strides[i], 0)
                            and not dep.is_indirect()
                        ):
                            dim[i].count_per_thread_broadcast += 1
                            dim[i].bytes_per_thread_broadcast += itemsize
                        else:
                            dim[i].count_per_thread_non_contiguous += 1
                            dim[i].bytes_per_thread_non_contiguous += itemsize
                            contiguous_or_broadcast[i] = False
                    numel += dep.get_numel()
                if len(deps) > 1:
                    # can't read more elements than exist in the buffer
                    numel = sympy.Min(numel, V.graph.get_numel(name))
                nbytes = numel * itemsize
                for i in range(ndim):
                    if contiguous_or_broadcast[i]:
                        dim[i].bytes_contiguous_or_broadcast += nbytes
                    else:
                        dim[i].bytes_non_contiguous += nbytes
                if any(contiguous_or_broadcast):
                    result.bytes_contiguous_or_broadcast += nbytes
                else:
                    result.bytes_non_contiguous += nbytes
        if len(result.loop) > 1:
            # the first loop represent the "outside of the loop" compute which could be long lived
            result.loop = [result.loop[0] + x for x in result.loop[1:]]
        return result


@dataclasses.dataclass
class StatsForKernelType:
    """Memory usage stats that are collected for both persistent and looped kernels"""

    reads: StatsForReadsOrWrites
    writes: StatsForReadsOrWrites
    memory: StatsForReadsOrWrites

    @classmethod
    def compute(
        cls, loops: list[MemoryEstimate], estimator: MemoryEstimator
    ) -> typing.Self:
        reads = StatsForReadsOrWrites.compute(
            [loop.reads for loop in loops], estimator.symbols
        )
        writes = StatsForReadsOrWrites.compute(
            [loop.writes for loop in loops], estimator.symbols
        )
        return cls(
            reads=reads,
            writes=writes,
            memory=reads + writes,
        )


@dataclasses.dataclass
class MemoryStats:
    """Memory usage stats collected for each generated kernel"""

    persistent: StatsForKernelType
    looped: StatsForKernelType

    def get(self, persistent: bool) -> StatsForKernelType:
        return self.persistent if persistent else self.looped

    @classmethod
    def compute(cls, estimator: MemoryEstimator) -> typing.Self:
        persistent = StatsForKernelType.compute([estimator.persistent], estimator)
        if len(estimator.loops) == 1 and not (
            estimator.outside_loop and estimator.loops[0]
        ):
            looped = persistent  # loops/persistent is the same in this common case
        else:
            looped = StatsForKernelType.compute(
                [estimator.outside_loop, *estimator.loops], estimator
            )
        return cls(
            persistent=persistent,
            looped=looped,
        )
