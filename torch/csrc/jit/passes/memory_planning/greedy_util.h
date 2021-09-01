#pragma once

#include <torch/csrc/jit/passes/memory_planning.h>

namespace torch {
namespace jit {

using OffsetFinder = int64_t(LiveRange, int64_t, std::vector<MemAllocation>);

OffsetFinder findOffsetWithSmallestGap;

OffsetFinder findFirstOffset;

void makeAllocation(
    std::vector<MemAllocation>& ordered_allocations,
    std::unordered_map<LiveRange, int64_t, live_range_hash>
        managed_live_ranges,
    LiveRange unalloced_lvr,
    OffsetFinder findOffset);
} // namespace jit
} // namespace torch