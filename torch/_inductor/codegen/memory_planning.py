# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import dataclasses
import itertools
import math
import os
import pprint
from typing import Any, Protocol, TYPE_CHECKING

import sympy
import torch
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import Max

from .. import config
from ..utils import (
    _align,
    align,
    ALIGN_BYTES,
    cache_on_self,
    CachedMethod,
    IndentedBuffer,
)
from ..virtualized import V
from .wrapper import (
    AllocateLine,
    BufferLike,
    FreeIfNotReusedLine,
    MemoryPlanningLine,
    NullLine,
    ReuseLine,
)


def _cudagraph_slab_cache_enabled(wrapper: Any) -> bool:
    """The AOTI cuda-graph runtime captures partitions whose kernels bake in the
    address of every reinterpret_tensor view INTO the memory_planning slab.
    If the slab is re-allocated each forward (the default empty_strided), those
    captured addresses go stale and a replay reads/writes the wrong memory ->
    illegal memory access or silent garbage. Caching the slab per-shape in the
    wrapper makes the base address stable across forwards of the same shape,
    so the captured partitions stay valid. Gated to AOTI cpp wrapper + cuda-graph;
    Python/CPU paths are unaffected.
    """
    if not config.aot_inductor.enable_cuda_graph:
        return False
    if not V.graph.aot_mode:
        return False
    # cpp_wrapper_cpu.CppWrapperCpu (inherited by CppWrapperGpu) defines
    # make_allocation; the Python PythonWrapperCodegen path has a different API.
    return hasattr(wrapper, "codegen_int_array_var")


if TYPE_CHECKING:
    from collections.abc import Iterable

    import sympy


@dataclasses.dataclass
class LiveRange:
    """
    A range where a given tensor is live.  Begin and end are both counters
    representing points in the program of grouped memory operations.
    Begin is inclusive, end is exclusive.

    Invariant: begin <= end
    """

    begin: float  # int | +/-inf
    end: float  # int | +/-inf

    def contains(self, other: LiveRange):
        """Is other entirely within self"""
        return self.begin <= other.begin and other.end <= self.end

    def join(self, other: LiveRange):
        """Combine two ranges using a union operation"""
        return LiveRange(min(self.begin, other.begin), max(self.end, other.end))

    def __len__(self):
        return self.end - self.begin


class LiveRanges:
    """
    A collection of LiveRange regions, allowing for non-contiguous
    live regions.

    Invariant: LiveRanges.ranges is in sorted order and non-overlapping
    """

    def __init__(self, ranges: Iterable[LiveRange]):
        ranges = [*sorted(ranges, key=lambda x: x.begin)]
        self.ranges = ranges[:1]
        for r in ranges[1:]:
            if self.ranges[-1].begin > r.begin:
                raise AssertionError("ranges must be sorted by begin")
            if self.ranges[-1].end >= r.begin:
                self.ranges[-1] = LiveRange.join(self.ranges[-1], r)
            else:
                self.ranges.append(r)

    def overlaps(self, other: LiveRanges):
        """Check if any pair of ranges in self and other overlap"""
        left = collections.deque(self.ranges)
        right = collections.deque(other.ranges)
        while left and right:
            if left[0].begin > right[0].begin:
                left, right = right, left
            if left[0].begin > right[0].begin:
                raise AssertionError("left should begin no later than right")
            if left[0].end > right[0].begin:
                return True
            left.popleft()
        return False

    @property
    def begin(self):
        return self.ranges[0].begin

    @property
    def end(self):
        return self.ranges[-1].end

    def __repr__(self):
        return f"{self.__class__.__name__}([{', '.join(map(repr, self.ranges))}])"


class AllocationTreeNode:
    """
    Abstract base class for nodes in allocation pool.
    """

    def allocate(self, block: Allocation, is_last: bool) -> bool:
        """
        Try to assign block to a memory location in this bool.  Return True if
        an assignment was made.
        """
        return False

    def get_live_ranges(self) -> LiveRanges:
        """Aggregate LiveRanges for all objects below this in tree"""
        raise NotImplementedError

    def get_size_hint(self) -> int:
        """Number of bytes used for example inputs"""
        raise NotImplementedError

    def get_symbolic_size(self) -> sympy.Expr:
        """Number of bytes needed at runtime"""
        raise NotImplementedError

    def finalize(self, pool, offset) -> AllocationTreeNode:
        """Called after all allocations have been made"""
        return self

    def is_empty(self):
        return False


@dataclasses.dataclass
class Allocation(AllocationTreeNode):
    """
    Represents memory allocated to a given node in the allocation pool.
    """

    node: BufferLike
    live_range: LiveRange
    size_hint: int
    symbolic_size: sympy.Expr
    allocated: bool = False
    pool: AllocationPool | None = None
    offset: sympy.Expr | None = None
    earliest_available: float | None = None

    def __post_init__(self) -> None:
        has_unbacked_sym = False
        for s in self.node.get_layout().size:
            if free_unbacked_symbols(s):
                has_unbacked_sym = True
                break

        if has_unbacked_sym:
            self.earliest_available = self.get_live_ranges().begin

    @property
    def device(self):
        return self.node.get_device()

    def get_live_ranges(self):
        return LiveRanges([self.live_range])

    def get_size_hint(self):
        return self.size_hint

    def get_symbolic_size(self):
        return self.symbolic_size

    def mark_allocated(self):
        if self.allocated:
            raise AssertionError("block already allocated")
        self.allocated = True

    def finalize(self, pool, offset):
        if not (self.pool is None and self.offset is None):
            raise AssertionError("block already finalized")
        self.pool = pool
        self.offset = offset
        return self

    def codegen_alloc_from_pool(self, wrapper):
        if not self.pool:
            raise AssertionError("block has no pool assigned")
        node = self.node
        shape = tuple(node.get_size())
        stride = tuple(node.get_stride())
        return wrapper.codegen_alloc_from_pool(
            self.pool.name, self.offset, node.get_dtype(), shape, stride
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"node={self.node.get_name()}, "
            f"live_range={self.live_range}, "
            f"size_hint={self.size_hint}, "
            f"symbolic_size={self.symbolic_size}, "
            f"pool={self.pool.name if self.pool else None}, "
            f"offset={self.offset})"
        )

    def get_earliest_available(self):
        return self.earliest_available


@dataclasses.dataclass
class Empty(AllocationTreeNode):
    """
    Placeholder to represent empty space in the allocation pool.
    Only exists to get the size_hint correct in parent nodes.
    """

    size_hint: int

    def get_live_ranges(self):
        return LiveRanges([])

    def get_size_hint(self):
        return self.size_hint

    def get_symbolic_size(self):
        return 0

    def is_empty(self):
        return True


class MemorySplitProtocol(Protocol):
    get_live_ranges: CachedMethod[[], LiveRanges]
    get_size_hint: CachedMethod[[], int]
    get_symbolic_size: CachedMethod[[], sympy.Expr]

    def _allocate(self, block: Allocation, is_last: bool) -> bool: ...


class ClearCacheOnAllocateMixin(MemorySplitProtocol):
    """
    Helper to assist in caching get_live_ranges, get_size_hint, and
    get_symbolic_size.
    """

    def allocate(self, block: Allocation, is_last: bool):
        is_allocated = self._allocate(block, is_last)
        if is_allocated:
            self.clear_cache()
        return is_allocated

    def clear_cache(self):
        self.get_live_ranges.clear_cache(self)
        self.get_size_hint.clear_cache(self)
        self.get_symbolic_size.clear_cache(self)


@dataclasses.dataclass
class TemporalSplit(ClearCacheOnAllocateMixin, AllocationTreeNode):
    """
    Contains a list of allocations not overlapping in LiveRanges.

    Invariant: no pair (a,b) in self.allocations will have:
         a.get_live_ranges().overlaps(b.get_live_ranges())
    """

    allocations: list[AllocationTreeNode]

    def _allocate(self, block: Allocation, is_last: bool):
        slot_size = self.get_size_hint()
        block_size = block.get_size_hint()
        if not is_last and block_size > slot_size:
            return False  # doesn't fit

        block_live = block.get_live_ranges()
        overlapping = [
            s for s in self.allocations if s.get_live_ranges().overlaps(block_live)
        ]
        if len(overlapping) > 1:
            # TODO(jansel): we could try harder here by merging overlapping in space
            return False
        elif len(overlapping) == 1:
            return overlapping[0].allocate(block, is_last)
        else:
            block.mark_allocated()

            if len(self.allocations) == 1 and isinstance(self.allocations[-1], Empty):
                self.allocations.pop()

            if slot_size == block_size:
                # perfect fit
                self.allocations.append(block)
            elif slot_size > block_size:
                self.allocations.append(
                    SpatialSplit.create(block, slot_size - block_size)
                )
            else:  # grow this allocation
                if not is_last:
                    raise AssertionError("can only grow allocation when is_last")
                self.allocations = [
                    *(
                        SpatialSplit.create(a, block_size - slot_size)
                        for a in self.allocations
                    ),
                    block,
                ]
            return True

    @cache_on_self
    def get_live_ranges(self) -> LiveRanges:
        return LiveRanges(
            itertools.chain.from_iterable(
                x.get_live_ranges().ranges for x in self.allocations
            )
        )

    @cache_on_self
    def get_size_hint(self) -> int:
        if not self.allocations:
            return 0
        return max(x.get_size_hint() for x in self.allocations)

    @cache_on_self
    def get_symbolic_size(self) -> sympy.Expr:
        if not self.allocations:
            return 0  # type: ignore[return-value]
        return Max(*[x.get_symbolic_size() for x in self.allocations])

    def is_empty(self):
        return len(self.allocations) == 1 and self.allocations[0].is_empty()

    def finalize(self, pool, offset):
        self.allocations = [block.finalize(pool, offset) for block in self.allocations]
        self.clear_cache()
        if len(self.allocations) == 1:
            return self.allocations[0]
        return self


@dataclasses.dataclass
class SpatialSplit(ClearCacheOnAllocateMixin, AllocationTreeNode):
    """
    Contains two allocations, left and right, that do not overlap in space.
    Right will be allocated immediately after left in memory.
    """

    left: TemporalSplit
    right: TemporalSplit

    @staticmethod
    def create(left, extra_space):
        if not isinstance(left, AllocationTreeNode):
            raise AssertionError(f"expected AllocationTreeNode, got {type(left)}")
        if not (isinstance(extra_space, int) and extra_space >= 1):
            raise AssertionError(
                f"expected positive int extra_space, got {extra_space}"
            )
        return SpatialSplit(TemporalSplit([left]), TemporalSplit([Empty(extra_space)]))

    def _allocate(self, block: Allocation, is_last: bool):
        return self.left.allocate(block, False) or self.right.allocate(block, is_last)

    @cache_on_self
    def get_live_ranges(self):
        return LiveRanges(
            itertools.chain(
                self.left.get_live_ranges().ranges, self.right.get_live_ranges().ranges
            )
        )

    @cache_on_self
    def get_size_hint(self) -> int:
        return _align(self.left.get_size_hint()) + self.right.get_size_hint()

    @cache_on_self
    def get_symbolic_size(self) -> sympy.Expr:
        return align(self.left.get_symbolic_size()) + self.right.get_symbolic_size()

    def finalize(self, pool, offset):
        self.left = self.left.finalize(pool, offset)
        self.right = self.right.finalize(
            pool, offset + align(self.left.get_symbolic_size())
        )
        self.clear_cache()
        if self.right.is_empty():
            return self.left
        return self


@dataclasses.dataclass
class AllocationPool:
    """
    Represents a pool of allocations that will be generated by a single
    call to torch.empty.
    """

    device: torch.device
    root: TemporalSplit
    can_expand: bool = True
    restrict_live_range: LiveRange | None = None
    name: str | None = None
    names_to_del: list[str] = dataclasses.field(default_factory=list)
    creation_cache: dict[str, str] = dataclasses.field(default_factory=dict)
    # Explicit cudagraph-slab-cache id, folded into the per-instance cache key so
    # each pool's cached slab stays disjoint. None -> derive from the pool name
    # (pool0/pool1/... numeric path). Set explicitly for the cross-partition
    # handoff pools, whose names are not poolN, to a reserved collision-free id.
    pool_id: int | None = None

    def __post_init__(self) -> None:
        for block in self.root.allocations:
            if isinstance(block, Allocation):
                self.update_restrict_live_range(block)

    def allocate(self, block: Allocation, is_last: bool):
        if (
            self.restrict_live_range is not None
            and not self.restrict_live_range.contains(block.live_range)
        ):
            return False

        block_earliest_available = block.get_earliest_available()
        pool_begin = self.root.get_live_ranges().begin
        if block_earliest_available and block_earliest_available > pool_begin:
            return False

        is_last = self.can_expand and is_last
        if self.root.allocate(block, is_last):
            self.update_restrict_live_range(block)
            return True

        if is_last:
            return self.allocate_at_end(block)

        return False

    def update_restrict_live_range(self, block: Allocation):
        if block_earliest_available := block.get_earliest_available():
            if self.restrict_live_range is None:
                self.restrict_live_range = LiveRange(
                    block_earliest_available, float("inf")
                )
            else:
                self.restrict_live_range = LiveRange(
                    min(self.restrict_live_range.begin, block_earliest_available),
                    self.restrict_live_range.end,
                )

    def allocate_at_end(self, block):
        block.mark_allocated()
        self.root = TemporalSplit([SpatialSplit(self.root, TemporalSplit([block]))])
        self.update_restrict_live_range(block)
        return True

    def finalize(self, name):
        if self.name:
            raise AssertionError("pool already finalized")
        self.name = name
        self.names_to_del.append(name)
        self.root.finalize(self, 0)

    def codegen_create(self, wrapper, code: IndentedBuffer):
        if not self.name:
            raise AssertionError("pool must be finalized before codegen_create")
        nbytes = self.root.get_symbolic_size()
        # Under AOTI cuda-graph the slab base address must be stable across
        # forwards so captured partitions' reinterpret_tensor views (which bake
        # the slab address at capture time) stay valid. Route the slab allocation
        # through a per-instance cached slab. See _cudagraph_slab_cache_enabled.
        if _cudagraph_slab_cache_enabled(wrapper):
            for block in self.root.allocations:
                if (
                    isinstance(block, Allocation)
                    and nbytes == block.get_symbolic_size()
                ):
                    node = block.node
                    self._codegen_create_cudagraph_cached(
                        wrapper,
                        code,
                        dtype=node.get_dtype(),
                        shape=tuple(node.get_size()),
                        stride=tuple(node.get_stride()),
                    )
                    return
            self._codegen_create_cudagraph_cached(
                wrapper,
                code,
                dtype=torch.uint8,
                shape=(nbytes,),
                stride=(1,),
            )
            return
        for block in self.root.allocations:
            if isinstance(block, Allocation) and nbytes == block.get_symbolic_size():
                node = block.node
                code.writeline(
                    wrapper.make_allocation(
                        self.name,
                        device=self.device,
                        dtype=node.get_dtype(),
                        shape=tuple(node.get_size()),
                        stride=tuple(node.get_stride()),
                    )
                )
                return
        else:
            code.writeline(
                wrapper.make_allocation(
                    self.name,
                    device=self.device,
                    dtype=torch.uint8,
                    shape=(nbytes,),
                    stride=(1,),
                )
            )

    def _codegen_create_cudagraph_cached(
        self, wrapper, code: IndentedBuffer, dtype, shape, stride
    ):
        """Emit the persistent per-shape slab. The cache is the per-instance
        member `this->cudagraph_slabs_` (an
        unordered_map<int64_t, RAIIAtenTensorHandle> on AOTInductorModel) that
        owns the allocation; the local `pool<i>` is a non-owning view at the
        cached base address, so the existing AllocFromPoolLine codegen (which
        calls `aoti_torch__alloc_from_pool(pool<i>, ...)`) is unchanged and free
        of per-replay re-allocation. The member is owned by the AOTInductorModel
        instance and freed with it (RAII), matching the per-instance scope of
        `this->cudagraph_mgr_` -- concurrent instances in a model_container get
        isolated slabs. The single member is shared by every pool, so the cache
        key folds a stable per-pool id into the high bits (see below) to keep
        pools disjoint; under single-max-slab the per-pool key is otherwise a
        constant, so every shape of a pool reuses that pool's one max-sized slab.

        Single-max-slab: when the slab's shape/stride contains dynamic
        symbols, size the slab ONCE to those symbols' upper bounds (max batch)
        and share it across all shapes via a constant cache key. Small shapes use
        a prefix of the max slab; the per-buffer offset views are unchanged and
        fit inside the max slab. Every contributing dynamic symbol must have a
        finite upper bound; an unbounded symbol is a compile-time error (see
        below). Static slabs (no dynamic symbol) keep the constant cache key
        directly.
        """
        # Collect the dynamic-symbol names referenced by the slab's shape OR
        # stride. Their presence decides whether we resize the slab to the
        # dynamic-shape upper bounds below; the slab packs many partitions so it
        # can depend on several (s13, s14, ...).
        sym_set: OrderedSet[sympy.Symbol] = OrderedSet()
        for expr in [*shape, *stride]:
            if isinstance(expr, sympy.Expr):
                for sym in expr.free_symbols:
                    if isinstance(sym, sympy.Symbol):
                        sym_set.add(sym)
        sym_list = sorted(sym_set, key=str)

        # Single-max-slab path: replace the symbolic shape/stride with
        # their dynamic-shape upper bounds (max batch) so ONE slab, keyed on a
        # constant, is shared across all shapes. Use the same bound_sympy
        # mechanism AOTI uses for runtime input upper-bound checks (see
        # cpp_wrapper_cpu.py). Every contributing symbol MUST have a finite upper
        # bound; an unbounded dim (math.isinf / sympy oo / int_oo) is a
        # compile-time error -- AOTI regional cuda-graph needs a fixed max slab
        # size, which an unbounded dim cannot provide.
        if sym_list:
            from torch.utils._sympy.numbers import int_oo
            from torch.utils._sympy.value_ranges import bound_sympy

            var_to_range = V.graph.sizevars.shape_env.var_to_range

            def _const_upper_bound(expr):
                if isinstance(expr, (int, sympy.Integer)):
                    return int(expr)
                if not isinstance(expr, sympy.Expr):
                    return None
                # The slab size/stride can contain Inductor's `align` sympy
                # Function (from SpatialSplit.get_symbolic_size / finalize), which
                # the ValueRanges interpreter behind bound_sympy has no handler
                # for -> KeyError. align(e) rounds e up to the next multiple of
                # ALIGN_BYTES, so align(e) <= e + (ALIGN_BYTES - 1); substitute
                # that boundable upper bound before bounding. Taking .upper then
                # still yields a valid (>= true max) slab dimension.
                expr = expr.replace(align, lambda e: e + (ALIGN_BYTES - 1))
                upper = bound_sympy(expr, var_to_range).upper
                if upper in (sympy.oo, int_oo) or math.isinf(upper):
                    return None
                return int(upper)

            max_shape = [_const_upper_bound(d) for d in shape]
            max_stride = [_const_upper_bound(s) for s in stride]
            if any(d is None for d in max_shape) or any(s is None for s in max_stride):
                raise RuntimeError(
                    "AOTI regional cuda-graph requires finite dynamic-shape upper "
                    "bounds; set them via the export Dim(max=...) / "
                    "dynamic_shapes_strategy. Slab shape="
                    f"{shape}, stride={stride} has an unbounded dynamic dimension."
                )
            # Size the slab to the max so a single slab is shared across all
            # shapes (the per-pool cache key below is a constant).
            shape = tuple(max_shape)
            stride = tuple(max_stride)
        # Under single-max-slab the per-shape key is always a constant: the slab
        # is sized to the dynamic-shape upper bounds and shared across shapes, so
        # only the per-pool id (folded in below) distinguishes cache slots.
        #
        # Memory planning is PER-PARTITION: each captured partition is codegen'd
        # through its OWN subgraph wrapper with its OWN MemoryPlanner, so pool
        # names RESTART per partition (every partition's first pool is "pool0",
        # pool_id 0). The slab cache (this->cudagraph_slabs_) is process-shared
        # across partition bodies, so without further disambiguation EVERY
        # partition's "pool0" maps to the SAME cache key (pool_id 0, key 0) and
        # thus the SAME physical slab. Two partitions then alias the same slab
        # offsets: a later partition's buffer clobbers an earlier partition's
        # slab-resident output that must survive to the end of the forward (model
        # outputs / handoffs), producing deterministic wrong outputs on every
        # forward (the captured graph itself is fine -- the memory is overwritten
        # by a downstream partition before the caller reads it). Per-subgraph
        # live-range analysis cannot see this: it only orders buffers WITHIN one
        # partition. So fold the owning partition id into the cache key here, in
        # the low 40 bits, giving each captured partition DISJOINT slabs. This is
        # the correctness baseline (no cross-partition memory reuse); cross-SHAPE
        # sharing within a partition is preserved (the key stays constant per
        # (partition, pool)). The id is partition_id + 1 so partition 0's slab
        # never collides with a non-subgraph (outer-scope) pool whose key low
        # bits are 0. Handoff pools are created in the OUTER wrapper (no
        # subgraph_name) and are already disambiguated by an explicit reserved
        # self.pool_id, so they keep key low bits 0.
        subgraph_name = getattr(wrapper, "subgraph_name", None)
        partition_key = 0
        if isinstance(subgraph_name, str) and subgraph_name.startswith("partition_"):
            partition_suffix = subgraph_name[len("partition_") :]
            if partition_suffix.isdigit():
                partition_key = int(partition_suffix) + 1
        key_expr = f"{partition_key}LL"
        shape_log_expr = '""'

        # Reuse the existing cpp wrapper helpers for size/stride/dtype/device
        # rendering -- this keeps the slab's ABI call identical to the
        # non-cached path (same aoti_torch_empty_strided signature).
        device_str = wrapper.codegen_device(self.device)
        dtype_code = wrapper.codegen_dtype(dtype)
        size_str = wrapper.codegen_shape_tuple(shape)
        stride_str = wrapper.codegen_shape_tuple(stride)
        device_type, device_id = device_str.split(",")
        device_idx = "this->device_idx_" if V.graph.aot_mode else device_id

        size_array_var = wrapper.codegen_int_array_var(
            size_str,
            wrapper.wrapper_call.writeline,
            known_statically=wrapper.is_statically_known_list_of_ints(shape),
            graph=wrapper.get_codegened_graph(),
        )
        stride_array_var = wrapper.codegen_int_array_var(
            stride_str,
            wrapper.wrapper_call.writeline,
            known_statically=wrapper.is_statically_known_list_of_ints(stride),
            graph=wrapper.get_codegened_graph(),
        )
        ndim = str(len(shape))

        pool_name = self.name
        cache_member = "this->cudagraph_slabs_"
        owner_name = f"{pool_name}_owner_handle"
        data_ptr_name = f"{pool_name}_data_ptr"
        view_name = f"{pool_name}_view_handle"
        key_name = f"{pool_name}_shape_key"
        # On only for a truthy AOTI_CGTREE_DEBUG. Unset/empty/"0" is off so a
        # production run that exports AOTI_CGTREE_DEBUG=0 (e.g. the scorecard
        # harness) does not codegen the per-forward [smp-bind] address audit.
        debug = os.environ.get("AOTI_CGTREE_DEBUG", "") not in ("", "0")

        # Stable per-pool id, folded into the cache key so the single shared
        # member keeps each pool's slabs disjoint. An explicit self.pool_id
        # (cross-partition handoff pools) wins; otherwise pools are named "pool0",
        # "pool1", ...; parse the trailing integer. Fall back to a hash bucket if
        # the name ever changes shape (keeps the id deterministic + bounded).
        pool_suffix = pool_name[len("pool") :] if pool_name.startswith("pool") else ""
        if self.pool_id is not None:
            pool_id = self.pool_id
        elif pool_suffix.isdigit():
            pool_id = int(pool_suffix)
        else:
            pool_id = abs(hash(pool_name)) % (1 << 20)

        # Per-instance cache (this->cudagraph_slabs_) survives across forwards and
        # is owned by the AOTInductorModel instance (NOT process-static), matching
        # the per-instance scope of this->cudagraph_mgr_ so concurrent instances
        # in a model_container have isolated slabs. The key mirrors the
        # cudagraph_tree.h encode() packing: the pool id occupies the high bits
        # ((pool_id) << 40) and the low 40 bits (key_expr & 0xFFFFFFFFFFLL) hold
        # the owning partition id (partition_key, set above) so a partition's
        # pool is disjoint from the same-named pool of every other partition.
        # pool_id and partition_key occupy disjoint bit ranges, so the key is
        # injective over (pool_id, partition_id); distinct pools never collide.
        code.writeline(
            f"int64_t {key_name} = "
            f"((int64_t){pool_id} << 40) ^ (({key_expr}) & 0xFFFFFFFFFFLL);"
        )
        code.writeline(f"auto {pool_name}_it = {cache_member}.find({key_name});")
        code.writeline(f"if ({pool_name}_it == {cache_member}.end()) {{")
        code.writeline(f"    AtenTensorHandle {owner_name};")
        code.writeline(
            f"    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided("
            f"{ndim}, {size_array_var}, {stride_array_var}, "
            f"{dtype_code}, {device_type}, {device_idx}, &{owner_name}));"
        )
        code.writeline(
            f"    auto {pool_name}_ins = {cache_member}.emplace("
            f"{key_name}, RAIIAtenTensorHandle({owner_name}));"
        )
        code.writeline(f"    {pool_name}_it = {pool_name}_ins.first;")
        if debug:
            code.writeline(
                f'    std::cerr << "[smp-bind] FIRST pool=" << "{pool_name}"'
                f' << " shape=" << {shape_log_expr}'
                f' << " addr=" << ({pool_name}_it->second.operator AtenTensorHandle())'
                f" << std::endl;"
            )
        code.writeline("}")
        # NOTE on the "largest-seen" defensive guard (peer's caveat): the slab
        # nbytes is a sympy formula in the dynamic symbols (s13/s14) so it is
        # deterministic for a fixed shape key -- the same key cannot produce
        # different slab sizes. Skipping the guard keeps the per-forward path
        # branch-free; if a future refactor ever decouples size from key,
        # re-add a `static unordered_map<int64_t,int64_t> shape_to_nbytes` check
        # and grow via empty_strided + std::move.
        # Audit: detect any drift between first-bind and reuse (must be impossible
        # with this construction, but guards future refactors). Cheap, gated on
        # AOTI_CGTREE_DEBUG -- zero cost when unset.
        code.writeline(f"void* {data_ptr_name} = nullptr;")
        code.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr("
            f"{pool_name}_it->second, &{data_ptr_name}));"
        )
        if debug:
            audit_map = f"{pool_name}_audit"
            code.writeline(f"static std::unordered_map<int64_t, void*> {audit_map};")
            code.writeline(f"auto {pool_name}_ait = {audit_map}.find({key_name});")
            code.writeline(f"if ({pool_name}_ait == {audit_map}.end()) {{")
            code.writeline(f"    {audit_map}.emplace({key_name}, {data_ptr_name});")
            code.writeline("} else {")
            code.writeline(f"    if ({pool_name}_ait->second != {data_ptr_name}) {{")
            code.writeline(
                f'        std::cerr << "[smp-VIOLATION P1] pool=" << "{pool_name}"'
                f' << " shape=" << {shape_log_expr}'
                f' << " addr_now=" << {data_ptr_name}'
                f' << " addr_first=" << {pool_name}_ait->second'
                f" << std::endl;"
            )
            code.writeline("        std::abort();")
            code.writeline("    }")
            code.writeline("}")
            code.writeline(
                f'std::cerr << "[smp-bind] REUSE pool=" << "{pool_name}"'
                f' << " shape=" << {shape_log_expr}'
                f' << " addr=" << {data_ptr_name}'
                f" << std::endl;"
            )
        # Expose the cached OWNER handle directly as a raw (borrowed) AtenTensorHandle
        # named after the pool. The downstream codegen (`AllocFromPoolLine`) emits
        # `aoti_torch__alloc_from_pool(pool<i>, ...)`, which accepts any
        # AtenTensorHandle; an AtenTensorHandle is just a pointer to TensorImpl
        # and is implicitly convertible to/from raw, so the existing call sites
        # compile unchanged. We deliberately do NOT wrap in a local
        # RAIIAtenTensorHandle: the per-instance member (this->cudagraph_slabs_)
        # owns the handle for the model's lifetime, and a local RAII destructor
        # would try to free it on `codegen_destroy` -> `pool<i>.reset()`.
        # `codegen_destroy` is overridden below to omit the pool from
        # `names_to_del` so no `.reset()` is emitted.
        code.writeline(
            f"AtenTensorHandle {pool_name} = "
            f"static_cast<AtenTensorHandle>({pool_name}_it->second);"
        )

    def codegen_destroy(self, wrapper, code: IndentedBuffer):
        # The cg-cached pool slab is owned by the per-instance member
        # this->cudagraph_slabs_; the local `pool<i>` is a raw borrowed handle, so
        # emitting `pool<i>.reset()` would not compile and would also try to free a
        # member-owned handle. Drop the pool name from the destroy list when the
        # cached path was used; the slab is freed by the member's RAII at model
        # destruction.
        #
        # Additionally, when codegenning a captured cuda-graph partition body
        # (cpp_wrapper_gpu.codegen_partition_call has set partition_signatures
        # with skip_cudagraph=False), the partition body ends with
        # `partition_outputs[i] = bufXXX.release();` for each output handed off
        # to the cg-tree runtime. The default pool-last-use destroy emits
        # `bufXXX.reset();` for ALL pool-allocated names INCLUDING bufXXX,
        # nullifying the handle before `.release()` runs and SIGSEGV'ing in
        # `cuda_graph_capture_meta -> aoti_torch_get_data_ptr(nullptr)`. So
        # drop subgraph-output names from the destroy list -- they're still
        # owned via the outer-scope handoff and freed by the caller's RAII at
        # end of wrapper_call.
        if _cudagraph_slab_cache_enabled(wrapper):
            # Emit NO per-buffer reset. Every pooled buffer is a view into the
            # member-cached slab; any early reset risks nulling a handle still
            # needed across a partition/scope boundary -- a partition output handed
            # to the cg-tree (`= bufXXX.release()` next line -> capture_meta
            # nullptr) or a boundary buffer that is a LATER captured partition's
            # input, reset by an earlier scope before that partition records (->
            # clone_preserve_strides nullptr, cudagraph_tree.h:351). Output-name
            # exclusion can't cover inputs produced by another scope. Views are
            # freed by local RAII at scope exit; the slab is fixed and owned by
            # this->cudagraph_slabs_, and in-order cuda-graph replay keeps reuse
            # sequence-correct.
            return
        code.writeline(wrapper.make_free_by_names(self.names_to_del))

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


@dataclasses.dataclass
class AllocationPools:
    """
    Collection of many AllocationPool objects grouped by device.
    """

    device_to_pools: dict[torch.device, list[AllocationPool]] = dataclasses.field(
        default_factory=dict
    )

    def get_pools(self, block):
        if block.device not in self.device_to_pools:
            self.device_to_pools[block.device] = []
        return self.device_to_pools[block.device]

    def allocate(self, block: Allocation):
        pools = self.get_pools(block)

        for pool in pools:
            if pool.allocate(block, is_last=pool is pools[-1]):
                return

        # everything is full, make a new pool
        pools.append(
            AllocationPool(
                block.device,
                TemporalSplit([block]),
                can_expand=config.memory_pool != "none",
            )
        )
        block.mark_allocated()

    def allocate_output(self, block: Allocation):
        """Outputs get different pools so memory gets freed properly"""
        pools = self.get_pools(block)
        if pools and config.memory_pool in ("outputs", "combined"):
            pools[-1].allocate_at_end(block)
        else:
            # create a new pool
            block.mark_allocated()
            pools.append(
                AllocationPool(
                    block.device,
                    TemporalSplit([block]),
                    can_expand=config.memory_pool == "combined",
                )
            )

    def finalize(self):
        """Called at the end of allocation process"""
        for i, pool in enumerate(
            itertools.chain.from_iterable(self.device_to_pools.values())
        ):
            pool.finalize(f"pool{i}")

    def pprint(self):
        for pool in itertools.chain.from_iterable(self.device_to_pools.values()):
            print()
            print(pool.name)
            print(pool.root.get_live_ranges())
            pprint.pprint(pool.root)


class BufferGroup:
    """
    Due to inplace reuse an allocated buffer can have many names.
    This tracks these collections of buffers sharing underlying memory.
    """

    def __init__(self, node: BufferLike):
        self.node = node
        self.names = [node.get_name()]
        self.is_output = False
        self.allocation: Allocation | None = None
        self.live_range = LiveRange(float("inf"), -float("inf"))

    def update_usage(self, timestep: int):
        """Expand self.live_range to include timestep"""
        self.live_range = LiveRange(
            min(timestep, self.live_range.begin),
            max(timestep, self.live_range.end),
        )

    def sym_nbytes(self):
        return self.node.get_layout().storage_size() * self.node.get_dtype().itemsize

    def make_allocation(self):
        if self.allocation:
            raise AssertionError("multiple allocations")
        if not isinstance(self.live_range.begin, int):
            raise AssertionError("live ranges not computed")
        nbytes = self.sym_nbytes()
        # For now, fallback value will be used if we encounter an unbacked SymInt. The longer-term plan is to have
        # size_hint() use better heuristics for unbackeds, at which point the fallback value will be ignored.
        size_hint = V.graph.sizevars.optimization_hint(nbytes, fallback=64)
        self.allocation = Allocation(
            self.node,
            self.live_range,
            size_hint=size_hint,
            symbolic_size=nbytes,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.names!r}, is_output={self.is_output}, "
            f"live_range={self.live_range}"
        )


@dataclasses.dataclass
class PoolMemoryPlanningLine(MemoryPlanningLine):
    """Abstract base class for {Alloc,Dealloc}FromPoolLine"""

    group: BufferGroup
    timestep: int | None = None

    @property
    def node(self):
        return self.group.node


@dataclasses.dataclass
class AllocFromPoolLine(PoolMemoryPlanningLine):
    """Similar to AllocationLine, but takes memory from a pool"""

    is_first_pool_usage: bool = False

    def codegen(self, code: IndentedBuffer):
        allocation = self.group.allocation
        if not (allocation and allocation.pool):
            raise AssertionError("group must have an allocation with a pool")
        pool = allocation.pool
        name = self.node.get_name()

        # Cross-partition handoff producer-suppression + redirect. When this
        # buffer is a published cross-partition handoff, its storage lives in the
        # OUTER-owned per-symbol slab (cudagraph_handoff_pool_<sym>), in scope in
        # this producer body via the partition lambda's [&] capture. Bind the
        # name to a view into that slab at the published byte offset instead of
        # this subgraph's pool0; the existing generate_return .release() then
        # hands off the slab-resident view. pool0 is still created (for the
        # partition's other pooled buffers) but this handoff is NOT added to
        # pool0.names_to_del -- it is owned by the outer slab, not pool0. The
        # dict is empty (no redirect) unless plan_cudagraph_handoff_slab ran for
        # this graph, so non-cuda-graph codegen is unaffected.
        boundary_offsets = getattr(V.graph, "_cudagraph_boundary_offsets", None) or {}
        if name in boundary_offsets:
            if self.is_first_pool_usage:
                pool.codegen_create(self.wrapper, code)
            handoff_pool_name, handoff_offset = boundary_offsets[name]
            alloc_from_pool, lines_to_write = self.wrapper.codegen_alloc_from_pool(
                handoff_pool_name,
                handoff_offset,
                self.node.get_dtype(),
                tuple(self.node.get_size()),
                tuple(self.node.get_stride()),
            )
            code.writelines(lines_to_write)
            code.writeline(
                f"{self.wrapper.declare}{name} = {alloc_from_pool}{self.wrapper.ending}"
            )
            return

        if self.is_first_pool_usage:
            pool.codegen_create(self.wrapper, code)

        pool.names_to_del.extend(self.group.names)
        alloc_from_pool, allocation_lines_to_write = allocation.codegen_alloc_from_pool(
            self.wrapper
        )
        code.writelines(allocation_lines_to_write)
        if alloc_from_pool in pool.creation_cache:
            code.writeline(
                self.wrapper.make_tensor_alias(
                    name, pool.creation_cache[alloc_from_pool], "alloc"
                )
            )
        else:
            pool.creation_cache[alloc_from_pool] = name
            code.writeline(
                f"{self.wrapper.declare}{name} = {alloc_from_pool}{self.wrapper.ending}"
            )


@dataclasses.dataclass
class DeallocFromPoolLine(PoolMemoryPlanningLine):
    """Similar to FreeIfNotReusedLine, but takes memory from a pool"""

    is_last_pool_usage: bool = False

    def codegen(self, code: IndentedBuffer):
        if self.is_last_pool_usage:
            if not (self.group.allocation and self.group.allocation.pool):
                raise AssertionError("group must have an allocation with a pool")
            self.group.allocation.pool.codegen_destroy(self.wrapper, code)


def _resolve_handoff_buffer(name: str) -> BufferLike | None:
    """Return the underlying ir.Buffer for a handoff name, or None.

    V.graph.try_get_buffer may return a TensorBox/StorageBox (MutableBox)
    wrapping the Buffer; peel the box so the synthetic Allocation reads the
    real layout. WorkspaceArg / TorchBindObject have no tensor layout and
    are not valid handoff buffers, so they resolve to None.
    """
    from .. import ir

    node = V.graph.try_get_buffer(name)
    while isinstance(node, ir.MutableBox):
        node = node.data
    if isinstance(node, ir.Buffer):
        return node
    return None


def _pack_handoff_pool(
    pool_name: str,
    named_allocations: list[tuple[str, Allocation]],
    pool_id: int,
) -> AllocationPool:
    """Pack one per-symbol group of handoff allocations into a single pool.

    Seeds the pool with the first block, then allocates the rest with
    is_last=True so a block that fits no existing slot grows the pool at the
    end; allocate_at_end is the fallback when allocate() refuses a block
    whose earliest_available (unbacked-symint shapes) is past the pool
    begin. finalize() assigns each Allocation its byte offset. pool_id is the
    reserved cudagraph-slab-cache id for this pool.
    """
    first_allocation = named_allocations[0][1]
    first_allocation.mark_allocated()
    pool = AllocationPool(
        first_allocation.device,
        TemporalSplit([first_allocation]),
        can_expand=config.memory_pool != "none",
        pool_id=pool_id,
    )
    for _name, allocation in named_allocations[1:]:
        if not pool.allocate(allocation, is_last=True):
            pool.allocate_at_end(allocation)
    pool.finalize(pool_name)
    return pool


def _cudagraph_handoff_slab_planning_enabled() -> bool:
    """Wrapper-free gate for the handoff slab planning, equivalent to
    _cudagraph_slab_cache_enabled but evaluated without a wrapper instance so it
    can run in the scheduler pre-pass (Phase 1).

    AOTI cpp-wrapper + GPU target, with cuda-graph + memory_planning on. Because
    this runs once on the outer graph in the scheduler pre-pass (NOT per subgraph
    wrapper), the original "only the outer wrapper" guard (subgraph_name is None)
    is satisfied implicitly. Off (any condition false) -> early return, leaving
    non-cuda-graph behavior unchanged.
    """
    from ..utils import is_gpu

    if not config.aot_inductor.enable_cuda_graph:
        return False
    if not config.memory_planning:
        return False
    if not (V.graph.aot_mode and V.graph.cpp_wrapper):
        return False
    return is_gpu(V.graph.device_type)


def plan_cudagraph_handoff_slab() -> None:
    """Hoist captured-partition handoff buffers into OUTER-owned, PER-SYMBOL slab
    regions.

    Memory planning is per-partition: each captured partition is codegen'd
    through its own subgraph wrapper with its own MemoryPlanner -> its own
    AllocationPools, so pool names restart per partition (pool0 in every
    partition). The outer wrapper sees a partition only as one
    codegen_partition_call line, so it never builds an Allocation for any
    cross-partition handoff buffer (a partition output consumed by a LATER
    partition). Those handoff buffers therefore live in the per-partition
    runtime pool, not the slab.

    PHASE ORDERING (why this is a scheduler pre-pass, not a wrapper plan pass):
    the two readers of the dicts published here both run in Phase 1
    (Scheduler.codegen): (a) the per-partition body's AllocFromPoolLine.codegen
    reads _cudagraph_boundary_offsets, and (b) the outer codegen_partition_call
    reads _cudagraph_handoff_pools. The outer wrapper's MemoryPlanner.plan runs
    in Phase 2 (wrapper_code.generate, after Scheduler.codegen). So this MUST be
    called from the scheduler pre-pass, before the partition codegen loop, or the
    dicts would still be empty when both readers run.

    This pass runs ONCE on the outer graph and builds dedicated AllocationPools
    for exactly the buffers in V.graph._cudagraph_handoff_liveness (populated by
    the scheduler pre-pass). These buffers have NO outer AllocateLine, so we
    construct synthetic BufferGroup/Allocation objects directly from each
    buffer's ir.Buffer layout (size/stride/dtype live on V.graph, independent of
    which partition emits the buffer).

    PER-SYMBOL POOLS (the key ABI decision): a handoff's byte offset is a sum of
    aligned sizes of the OTHER handoffs packed before it in the same pool. The
    producer partition body emits alloc_from_pool(pool, offset, ...) inline, and
    a partition is restricted to <=1 dynamic symbol in its C++ scope. So we group
    handoffs by the single dynamic symbol of their symbolic nbytes and give each
    group its own pool: every offset in a pool then references only that pool's
    one symbol, which is exactly the symbol live in any producing partition's
    body. Static-size handoffs (no dynamic symbol) go to a separate static pool
    (int offsets). A handoff whose nbytes carries >=2 dynamic symbols cannot be
    made single-symbol-safe and is EXCLUDED (left on the runtime copy_in path ==
    current behavior).

    Within each pool, offsets are assigned by pid-interval overlap, NOT by
    timestep: each handoff's live_range is its [first_producer_pid ..
    last_consumer_pid] captured-partition execution interval, so two handoffs
    share a slot iff their pid intervals are disjoint. There is no global
    timestep axis (per-partition planners restart timestep at 0), so we bypass
    compute_live_ranges and feed the pid-interval live_range straight into the
    existing AllocationPool / TemporalSplit / SpatialSplit packer; pool.finalize()
    then assigns each Allocation a byte offset via Allocation.finalize(pool,
    offset).

    The result is exposed as
    V.graph._cudagraph_boundary_offsets[name] = (per_symbol_pool_name, offset),
    where per_symbol_pool_name is the single stable pool name for that symbol
    (resolving the per-partition pool0 ambiguity) and offset is the byte offset
    into that pool (int for static handoffs, a sympy.Expr in that pool's single
    symbol otherwise).

    Gated (via _cudagraph_handoff_slab_planning_enabled) on AOTI cpp-wrapper +
    GPU + cuda-graph + memory_planning; when off this function early-returns and
    leaves non-cuda-graph behavior unchanged.
    """
    if not _cudagraph_handoff_slab_planning_enabled():
        return

    handoff_liveness: dict[str, tuple[int, int]] = (
        getattr(V.graph, "_cudagraph_handoff_liveness", None) or {}
    )
    if not handoff_liveness:
        V.graph._cudagraph_boundary_offsets = {}
        return

    # Group handoffs by the single dynamic symbol of their symbolic nbytes.
    # sym_key is "" for static (no dynamic symbol) -> the static pool.
    groups_by_sym: dict[str, list[tuple[str, Allocation]]] = {}
    for name, (producer_pid, last_consumer_pid) in handoff_liveness.items():
        node = _resolve_handoff_buffer(name)
        if node is None:
            # No materialized ir.Buffer (e.g. unexpected alias/view). Skip
            # slab placement; the codegen side keeps such a buffer on the
            # existing copy_in path.
            continue
        group = BufferGroup(node)
        syms = sorted(
            (s for s in group.sym_nbytes().free_symbols if isinstance(s, sympy.Symbol)),
            key=str,
        )
        if len(syms) >= 2:
            # Offsets in a shared pool would reference >1 symbol, but a
            # producer partition body has only its single symbol in scope.
            # Cannot place safely -> fall back to copy_in (current behavior).
            continue
        sym_key = str(syms[0]) if syms else ""
        # pid interval [producer .. last_consumer], both inclusive. LiveRange
        # end is exclusive, so +1 makes a handoff consumed at pid K and one
        # produced at pid K correctly overlap (both live during pid K's run)
        # and thus get disjoint slots, while truly disjoint intervals share.
        group.live_range = LiveRange(producer_pid, last_consumer_pid + 1)
        group.make_allocation()
        if group.allocation is None:
            raise AssertionError("handoff group has no allocation")
        groups_by_sym.setdefault(sym_key, []).append((name, group.allocation))

    boundary_offsets: dict[str, tuple[str, Any]] = {}
    # Retain the finalized pools so the OUTER cpp wrapper can emit each
    # handoff slab once (via AllocationPool.codegen_create, reusing the
    # per-instance cached-slab path). Keyed by stable pool name.
    handoff_pools: dict[str, AllocationPool] = {}
    # Reserved cache-key id base for handoff pools. The slab-cache key packs the
    # pool id into the high bits as ((int64_t)pool_id << 40) (see
    # _codegen_create_cudagraph_cached and cudagraph_tree.h encode()). With a
    # signed int64 key, the base must satisfy (base << 40) < 2**63: 1<<22 gives
    # (1<<22) << 40 == 2**62, in range and positive. (The earlier 1<<30 made
    # (1<<30) << 40 == 2**70, which overflows int64 and aliases to 0 -- handoff
    # pool index i would then collide with the ordinary pool{i} slab.) The base
    # also stays above the legacy hash-bucket range 1<<20 used for non-poolN
    # names, so handoff slabs never alias a per-partition pool0 slab in
    # this->cudagraph_slabs_.
    handoff_pool_id_base = 1 << 22
    for pool_index, (sym_key, named_allocations) in enumerate(
        sorted(groups_by_sym.items())
    ):
        # Keep base+index < 2**23 so (pool_id << 40) < 2**63 stays in signed
        # int64 range and distinct from the small ordinary pool ids 0..N and the
        # legacy 1<<20 hash bucket. 1<<17 distinct handoff pools is far beyond the
        # handful of dynamic symbols a model has.
        if pool_index >= (1 << 17):
            raise AssertionError(
                f"too many cudagraph handoff pools ({pool_index + 1}); "
                "pool id would exceed the reserved int64 key range"
            )
        pool_name = (
            f"cudagraph_handoff_pool_{sym_key}"
            if sym_key
            else "cudagraph_handoff_pool_static"
        )
        pool = _pack_handoff_pool(
            pool_name, named_allocations, handoff_pool_id_base + pool_index
        )
        # Compare by symbol NAME, not Symbol object: Inductor size symbols
        # carry assumptions (positive/integer) so a bare sympy.Symbol(name)
        # would not equal them and would spuriously fail the subset check.
        allowed = {sym_key} if sym_key else set()
        placed_any = False
        for name, allocation in named_allocations:
            if allocation.offset is None:
                raise AssertionError("handoff allocation not finalized")
            offset_syms = set()
            if isinstance(allocation.offset, sympy.Expr):
                offset_syms = {
                    str(s)
                    for s in allocation.offset.free_symbols
                    if isinstance(s, sympy.Symbol)
                }
            if not offset_syms.issubset(allowed):
                # Defensive: packing produced a cross-symbol offset despite
                # single-symbol grouping. Exclude rather than emit an
                # out-of-scope symbol in the producer body.
                continue
            boundary_offsets[name] = (pool_name, allocation.offset)
            placed_any = True
        if placed_any:
            handoff_pools[pool_name] = pool

    V.graph._cudagraph_boundary_offsets = boundary_offsets
    V.graph._cudagraph_handoff_pools = handoff_pools


@dataclasses.dataclass
class MemoryPlanner:
    """
    Coordination object to run memory planning passes during wrapper
    codegen.
    """

    wrapper: Any
    pools: AllocationPools = dataclasses.field(default_factory=AllocationPools)
    buffer_groups: list[BufferGroup] | None = None

    def plan(self, lines: list[Any]) -> list[Any]:
        """Call all the memory planning passes in sequence"""
        lines = [*lines]
        self.drop_removed_buffers(lines)
        self.convert_to_pool_lines(lines)
        self.compute_live_ranges(lines)
        self.allocate_groups()
        self.mark_first_last_usage(lines)
        # The cross-partition handoff slab is NOT planned here. Its readers (the
        # per-partition body AllocFromPoolLine redirect and the outer
        # codegen_partition_call) run in scheduler.codegen() (Phase 1), which
        # strictly precedes this outer wrapper plan (Phase 2). Planning is done in
        # the scheduler pre-pass instead -- see plan_cudagraph_handoff_slab and
        # its single call site in Scheduler._codegen_partitions.
        return lines

    def drop_removed_buffers(self, lines):
        """
        Replace any memory planning lines in V.graph.removed_buffers with NullLine
        """
        # drop any removed buffers
        for i, line in enumerate(lines):
            if isinstance(line, (AllocateLine, FreeIfNotReusedLine, ReuseLine)):
                if line.node.get_name() in V.graph.removed_buffers:
                    lines[i] = NullLine(self.wrapper)

    def compute_buffer_groups(self, lines):
        """
        Populates self.buffer_groups with BufferGroup objects that join
        allocations with common storage (due to inplace reuse) into a
        single object.
        """
        name_to_group = {}
        for line in lines:
            if isinstance(line, AllocateLine):
                name = line.node.get_name()
                if name in name_to_group:
                    raise AssertionError(f"duplicate allocation for {name}")
                name_to_group[name] = BufferGroup(line.node)
            elif isinstance(line, ReuseLine):
                old_name = line.node.get_name()
                new_name = line.reused_as.get_name()
                if new_name in name_to_group:
                    raise AssertionError(f"duplicate group for reused {new_name}")
                # TODO(jansel): we should support reusing buffers created via ExternKernelAlloc
                if old_name in name_to_group:
                    name_to_group[old_name].names.append(new_name)
                    name_to_group[new_name] = name_to_group[old_name]

        outputs = OrderedSet(V.graph.get_output_names())
        unique_groups = [*{id(g): g for g in name_to_group.values()}.values()]
        for group in unique_groups:
            group.is_output = any(x in outputs for x in group.names) or any(
                any(user.node.get_name() == "OUTPUT" for user in buf.users)
                for name in group.names
                if (buf := V.graph.scheduler.name_to_buf.get(name)) is not None
            )

        if self.buffer_groups is not None:
            raise AssertionError("buffer_groups already computed")
        self.buffer_groups = unique_groups
        return name_to_group

    def convert_to_pool_lines(self, lines):
        """
        Convert AllocateLine/FreeIfNotReusedLine/ReuseLine into their
        pool-based counterparts.
        """
        name_to_group = self.compute_buffer_groups(lines)
        for i, line in enumerate(lines):
            if isinstance(line, AllocateLine):
                if line.node.get_name() in name_to_group:
                    lines[i] = AllocFromPoolLine(
                        self.wrapper, name_to_group[line.node.get_name()]
                    )
            elif isinstance(line, FreeIfNotReusedLine):
                if line.is_reused:
                    raise AssertionError("expected line not to be reused")
                if line.node.get_name() in name_to_group:
                    lines[i] = DeallocFromPoolLine(
                        self.wrapper, name_to_group[line.node.get_name()]
                    )
            elif isinstance(line, ReuseLine):
                if line.node.get_name() in name_to_group:
                    line.delete_old = False

    def compute_live_ranges(self, lines):
        """Populate every BufferGroup.live_ranges field based on first/last usage"""
        timestep = 0
        worklist = collections.deque(lines)
        while worklist:
            if isinstance(worklist[0], MemoryPlanningLine):
                timestep += 1
                while worklist and isinstance(worklist[0], MemoryPlanningLine):
                    line = worklist.popleft()
                    if isinstance(line, PoolMemoryPlanningLine):
                        line.group.update_usage(timestep)
                        line.timestep = timestep
            else:
                worklist.popleft()

        timestep += 1
        if self.buffer_groups is None:
            raise AssertionError("buffer_groups not computed")
        for group in self.buffer_groups:
            if group.is_output:
                group.update_usage(timestep)

        # cg slab offset-reuse control. memory_planning packs buffers at shared
        # offsets via sequential live ranges, but cg capture/replay + eager-rerun
        # does not preserve those lifetimes, so a shared offset can be clobbered
        # across a partition boundary (deterministic out-N corruption). Forcing a
        # buffer's live range to overlap everything gives it a UNIQUE offset.
        # Protect ONLY cross-cg-boundary buffers (captured partition inputs +
        # outputs, _cudagraph_boundary_names): their slab offset must persist
        # until the captured partition replays, while intra-scope buffers reuse
        # safely (sequential within their own scope).
        if _cudagraph_slab_cache_enabled(self.wrapper):
            boundary = getattr(V.graph, "_cudagraph_boundary_names", None) or set()
            for group in self.buffer_groups:
                if any(nm in boundary for nm in group.names):
                    group.live_range = LiveRange(0, timestep + 1)

    def allocate_groups(self):
        """
        Assign every allocation to a specific location in a specific AllocationPool.
        """
        if config.memory_pool not in ("none", "intermediates", "outputs", "combined"):
            raise AssertionError(f"invalid memory_pool {config.memory_pool}")
        if self.buffer_groups is None:
            raise AssertionError("buffer_groups not computed")

        for group in self.buffer_groups:
            group.make_allocation()

        outputs: list[Allocation] = []
        intermediates: list[Allocation] = []
        for group in self.buffer_groups:
            if not group.allocation:
                raise AssertionError("group has no allocation")
            if group.is_output and config.memory_pool != "combined":
                outputs.append(group.allocation)
            else:
                intermediates.append(group.allocation)

        for block in sorted(
            outputs,
            key=lambda x: (
                x.size_hint,
                -len(x.live_range),
            ),
        ):
            self.pools.allocate_output(block)

        for block in sorted(
            intermediates,
            key=lambda x: (
                -x.size_hint,
                -len(x.live_range),
            ),
        ):
            self.pools.allocate(block)

        self.pools.finalize()

    def mark_first_last_usage(self, lines):
        """
        Populate the AllocFromPoolLine.is_first_pool_usage and
        DeallocFromPoolLine.is_last_pool_usage fields so that pools
        are created/destroyed.
        """
        seen = OrderedSet[AllocationPool]()
        for line in lines:
            if isinstance(line, AllocFromPoolLine):
                if not line.group.allocation:
                    raise AssertionError("group has no allocation")
                pool = line.group.allocation.pool
                if pool is None:
                    raise AssertionError("allocation has no pool")
                if pool not in seen:
                    line.is_first_pool_usage = True
                    seen.add(pool)

        seen = OrderedSet[AllocationPool]()
        for line in reversed(lines):
            if isinstance(line, DeallocFromPoolLine):
                if not line.group.allocation:
                    raise AssertionError("group has no allocation")
                pool = line.group.allocation.pool
                if pool is None:
                    raise AssertionError("allocation has no pool")
                if pool not in seen:
                    line.is_last_pool_usage = (
                        pool.root.get_live_ranges().end <= line.timestep
                    )
                    seen.add(pool)
