# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import dataclasses
import itertools
import pprint
from typing import Any, Dict, Iterable, List, Optional, Protocol

import sympy

import torch

from .. import config
from ..utils import _align, align, cache_on_self, CachedMethod, IndentedBuffer
from ..virtualized import V
from .wrapper import (
    AllocateLine,
    BufferLike,
    FreeIfNotReusedLine,
    MemoryPlanningLine,
    NullLine,
    ReuseLine,
)


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
            assert self.ranges[-1].begin <= r.begin
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
            assert left[0].begin <= right[0].begin
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
    pool: Optional[AllocationPool] = None
    offset: Optional[sympy.Expr] = None

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
        assert not self.allocated
        self.allocated = True

    def finalize(self, pool, offset):
        assert self.pool is None and self.offset is None
        self.pool = pool
        self.offset = offset
        return self

    def codegen_alloc_from_pool(self, wrapper):
        assert self.pool
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

    def _allocate(self, block: Allocation, is_last: bool) -> bool:
        ...


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

    allocations: List[AllocationTreeNode]

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
                assert is_last
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
        return sympy.Max(*[x.get_symbolic_size() for x in self.allocations])

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
        assert isinstance(left, AllocationTreeNode)
        assert isinstance(extra_space, int) and extra_space >= 1
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
    restrict_live_range: Optional[LiveRange] = None
    name: Optional[str] = None
    names_to_del: List[str] = dataclasses.field(default_factory=list)
    creation_cache: Dict[str, str] = dataclasses.field(default_factory=dict)

    def allocate(self, block: Allocation, is_last: bool):
        if self.restrict_live_range and not self.restrict_live_range.contains(
            block.live_range
        ):
            return False

        is_last = self.can_expand and is_last
        if self.root.allocate(block, is_last):
            return True

        if is_last:
            return self.allocate_at_end(block)

        return False

    def allocate_at_end(self, block):
        block.mark_allocated()
        self.root = TemporalSplit([SpatialSplit(self.root, TemporalSplit([block]))])
        return True

    def finalize(self, name):
        assert not self.name
        self.name = name
        self.names_to_del.append(name)
        self.root.finalize(self, 0)

    def codegen_create(self, wrapper, code: IndentedBuffer):
        assert self.name
        nbytes = self.root.get_symbolic_size()
        for block in self.root.allocations:
            if isinstance(block, Allocation) and nbytes == block.get_symbolic_size():
                # optimization: fuse first allocation and pool creation
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
                self.creation_cache[block.codegen_alloc_from_pool(wrapper)] = self.name
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

    def codegen_destroy(self, wrapper, code: IndentedBuffer):
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

    device_to_pools: Dict[torch.device, List[AllocationPool]] = dataclasses.field(
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
        self.allocation: Optional[Allocation] = None
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
        assert not self.allocation, "multiple allocations"
        assert isinstance(self.live_range.begin, int), "live ranges not computed"
        nbytes = self.sym_nbytes()
        # For now, fallback value will be used if we encounter an unbacked SymInt. The longer-term plan is to have
        # size_hint() use better heuristics for unbackeds, at which point the fallback value will be ignored.
        size_hint = V.graph.sizevars.size_hint(nbytes, fallback=64)
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
    timestep: Optional[int] = None

    @property
    def node(self):
        return self.group.node


@dataclasses.dataclass
class AllocFromPoolLine(PoolMemoryPlanningLine):
    """Similar to AllocationLine, but takes memory from a pool"""

    is_first_pool_usage: bool = False

    def codegen(self, code: IndentedBuffer):
        allocation = self.group.allocation
        assert allocation and allocation.pool
        pool = allocation.pool
        name = self.node.get_name()

        if self.is_first_pool_usage:
            pool.codegen_create(self.wrapper, code)

        pool.names_to_del.extend(self.group.names)
        alloc_from_pool = allocation.codegen_alloc_from_pool(self.wrapper)
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
            assert self.group.allocation and self.group.allocation.pool
            self.group.allocation.pool.codegen_destroy(self.wrapper, code)


@dataclasses.dataclass
class MemoryPlanner:
    """
    Coordination object to run memory planning passes during wrapper
    codegen.
    """

    wrapper: Any
    pools: AllocationPools = dataclasses.field(default_factory=AllocationPools)
    buffer_groups: Optional[List[BufferGroup]] = None

    def plan(self, lines: List[Any]) -> List[Any]:
        """Call all the memory planning passes in sequence"""
        lines = [*lines]
        self.drop_removed_buffers(lines)
        self.convert_to_pool_lines(lines)
        self.compute_live_ranges(lines)
        self.allocate_groups()
        self.mark_first_last_usage(lines)
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
                assert name not in name_to_group
                name_to_group[name] = BufferGroup(line.node)
            elif isinstance(line, ReuseLine):
                old_name = line.node.get_name()
                new_name = line.reused_as.get_name()
                assert new_name not in name_to_group
                # TODO(jansel): we should support reusing buffers created via ExternKernelAlloc
                if old_name in name_to_group:
                    name_to_group[old_name].names.append(new_name)
                    name_to_group[new_name] = name_to_group[old_name]

        outputs = set(V.graph.get_output_names())
        unique_groups = [*{id(g): g for g in name_to_group.values()}.values()]
        for group in unique_groups:
            group.is_output = any(x in outputs for x in group.names)

        assert self.buffer_groups is None
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
                assert not line.is_reused
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
        assert self.buffer_groups is not None
        for group in self.buffer_groups:
            if group.is_output:
                group.update_usage(timestep)

    def allocate_groups(self):
        """
        Assign every allocation to a specific location in a specific AllocationPool.
        """
        assert config.memory_pool in ("none", "intermediates", "outputs", "combined")
        assert self.buffer_groups is not None

        for group in self.buffer_groups:
            group.make_allocation()

        outputs: List[Allocation] = []
        intermediates: List[Allocation] = []
        for group in self.buffer_groups:
            assert group.allocation
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
        seen = set()
        for line in lines:
            if isinstance(line, AllocFromPoolLine):
                assert line.group.allocation
                pool = line.group.allocation.pool
                assert pool is not None
                if pool not in seen:
                    line.is_first_pool_usage = True
                    seen.add(pool)

        seen = set()
        for line in reversed(lines):
            if isinstance(line, DeallocFromPoolLine):
                assert line.group.allocation
                pool = line.group.allocation.pool
                assert pool is not None
                if pool not in seen:
                    line.is_last_pool_usage = (
                        pool.root.get_live_ranges().end <= line.timestep
                    )
                    seen.add(pool)
