# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import dataclasses
from typing import Any, List, Optional, TYPE_CHECKING

import torch

from ..virtualized import V
from .memory_planning import Allocation, AllocationPool, LiveRange, TemporalSplit
from .wrapper import (
    CommBufferAllocateLine,
    CommBufferFreeLine,
    CommBufferLine,
    NullLine,
)


if TYPE_CHECKING:
    from .. import ir
    from ..utils import IndentedBuffer


# NOTE [comm buffer planning]
#
# This file contains the memory planning logic for comm buffers. Compared to
# regular buffer planning, Inductor leverages the allocator's "persistent
# allocation" capability to meet the stringent requirements for registered
# buffer communication (see NOTE [lowering-time collective optimization] for
# details regarding the requirements).
#
# Comm buffer planning is a collaborative process between Inductor and the
# allocator. Inductor is responsible for planning comm buffers within a graph.
# For each (comm_buffer_type, group_name) pair, Inductor uses a single,
# persistently allocated pool to fulfill all comm buffer allocations. The
# allocator manages memory reuse across subgroups.
#
# In practice, comm buffer planning differs from regular buffer planning in the
# following ways:
#
# - Comm buffers can't use memory from regular pools, and non-comm buffers
# shouldn't use memory from comm buffer pools [1]. This means that (1) comm
# buffers are planned in isolation from regular buffers and (2) comm buffers
# don't participate in in-place reuse (this simplifies the logic).
# - To allow for memory reuse for persistent allocations across subgraphs,
# Inductor needs to "free" the pool before exiting each subgraph. This means
# that comm buffers cannot be graph outputs (this simplifies the logic).
# - Comm buffer pools are allocated with dedicated allocators.
# - Comm buffers for different (comm_buffer_type, group_name) pairs are planned
# separately in an isolated fashion.
#
# For comm buffer planning, we reuse most of the fundamental logic from regular
# buffer planning. To accommodate the above differences, we use the
# `CommBufferLine` hierarchy, which resembles the `MemoryPlanningLine`
# hierarchy, to represent comm buffer allocations at codegen time. We use the
# `CommBufferPlanner` to carry out comm buffer planning in an isolated fashion.
#
# Ideally, the comm buffer planning logic could be further consolidated with
# the existing buffer planning logic. For the time being, since the two code
# paths differ in maturity, we prefer isolation at the cost of some divergence.
#
# [1] Allowing non-comm buffers to use memory from comm buffer pools may
# unnecessarily increase the size of persistently allocated memory. In the
# future, we can optimize memory usage by performing comm buffer planning
# first, then letting regular buffer planning leverage the free live ranges
# from the comm buffer pools.
@dataclasses.dataclass
class CommBufferPlanningLine(CommBufferLine):
    block: Allocation
    timestep: Optional[int] = None


@dataclasses.dataclass
class CommBufferAllocFromPoolLine(CommBufferPlanningLine):
    is_first_pool_usage: bool = False

    def codegen(self, code: IndentedBuffer):
        pool = self.block.pool
        assert pool is not None
        name = self.node.get_name()

        if self.is_first_pool_usage:
            pool.codegen_create(self.wrapper, code)

        pool.names_to_del.append(self.node.get_name())
        alloc_from_pool = self.block.codegen_alloc_from_pool(self.wrapper)
        code.writeline(
            f"{self.wrapper.declare}{name} = {alloc_from_pool}{self.wrapper.ending}"
        )


@dataclasses.dataclass
class CommBufferDeallocFromPoolLine(CommBufferPlanningLine):
    is_last_pool_usage: bool = False

    def codegen(self, code: IndentedBuffer):
        if self.is_last_pool_usage:
            assert self.block.pool is not None
            self.block.pool.codegen_destroy(self.wrapper, code)


class CommBufferPool(AllocationPool):
    def __init__(
        self,
        device: torch.device,
        root: TemporalSplit,
        comm_buffer_type: ir.CommBufferType,
        group_name: str,
    ):
        super().__init__(device, root, can_expand=True)
        self.comm_buffer_type = comm_buffer_type
        self.group_name = group_name

    def codegen_create(self, wrapper, code: IndentedBuffer):
        assert self.name
        nbytes = self.root.get_symbolic_size()
        code.writeline(
            CommBufferAllocateLine.make_allocation_line(
                self.comm_buffer_type,
                self.group_name,
                wrapper,
                self.name,
                device=self.device,
                dtype=torch.uint8,
                shape=(nbytes,),
                stride=(1,),
            )
        )


@dataclasses.dataclass
class CommBufferPlanner:
    wrapper: Any
    comm_buffer_type: ir.CommBufferType
    group_name: str
    blocks: Optional[List[Allocation]] = None

    def plan(self, lines: List[Any]) -> List[Any]:
        lines = [*lines]
        self.drop_removed_buffers(lines)
        self.convert_to_pool_lines(lines)
        self.compute_live_ranges(lines)
        self.allocate()
        self.mark_first_last_usage(lines)
        return lines

    def drop_removed_buffers(self, lines):
        for i, line in enumerate(lines):
            if self.is_active_line(line, CommBufferAllocateLine) or self.is_active_line(
                line, CommBufferFreeLine
            ):
                if line.node.get_name() in V.graph.removed_buffers:
                    lines[i] = NullLine(self.wrapper)

    def convert_to_pool_lines(self, lines):
        name_to_block = {}
        for line in lines:
            if self.is_active_line(line, CommBufferAllocateLine):
                allocation = Allocation(
                    line.node,
                    LiveRange(float("inf"), -float("inf")),
                    size_hint=line.size,
                    symbolic_size=line.size,
                )
                name = line.node.get_name()
                assert name not in name_to_block
                name_to_block[name] = allocation
        self.blocks = list(name_to_block.values())

        for i, line in enumerate(lines):
            if self.is_active_line(line, CommBufferAllocateLine):
                name = line.node.get_name()
                lines[i] = CommBufferAllocFromPoolLine(
                    self.wrapper,
                    line.node,
                    name_to_block[name],
                )
            elif self.is_active_line(line, CommBufferFreeLine):
                name = line.node.get_name()
                assert name in name_to_block, (name, name_to_block)
                lines[i] = CommBufferDeallocFromPoolLine(
                    self.wrapper,
                    line.node,
                    name_to_block[name],
                )

    def compute_live_ranges(self, lines):
        timestep = 0
        worklist = collections.deque(lines)
        while worklist:
            if self.is_active_line(worklist[0], CommBufferPlanningLine):
                timestep += 1
                while worklist and self.is_active_line(
                    worklist[0], CommBufferPlanningLine
                ):
                    line = worklist.popleft()
                    line.block.live_range = LiveRange(
                        min(timestep, line.block.live_range.begin),
                        max(timestep, line.block.live_range.end),
                    )
                    line.timestep = timestep
            else:
                worklist.popleft()

    def allocate(self):
        assert self.blocks is not None
        self.blocks = sorted(
            self.blocks,
            key=lambda x: (
                -x.size_hint,
                -len(x.live_range),
            ),
        )

        pool = CommBufferPool(
            self.blocks[0].device,
            TemporalSplit([self.blocks[0]]),
            self.comm_buffer_type,
            self.group_name,
        )
        for block in self.blocks[1:]:
            pool.allocate(block, is_last=True)

        pool.finalize(name=f"{self.comm_buffer_type.value}_pool_{self.group_name}")

    def mark_first_last_usage(self, lines):
        for line in lines:
            if self.is_active_line(line, CommBufferAllocFromPoolLine):
                pool = line.block.pool
                assert pool is not None
                line.is_first_pool_usage = True
                break

        for line in reversed(lines):
            if self.is_active_line(line, CommBufferDeallocFromPoolLine):
                pool = line.block.pool
                assert pool is not None
                line.is_last_pool_usage = (
                    pool.root.get_live_ranges().end <= line.timestep
                )
                break

    def is_active_line(self, line, type):
        return isinstance(line, type) and (
            line.comm_buffer_type == self.comm_buffer_type
            and line.group_name == self.group_name
        )
