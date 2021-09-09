#pragma once

#include <torch/csrc/jit/passes/memory_planning.h>

namespace torch {
namespace jit {

using OffsetFinder =
    size_t(UniqueLiveRange, size_t, std::vector<MemAllocation>);

OffsetFinder findOffsetWithSmallestGap;

OffsetFinder findFirstOffset;

MemAllocation makeAllocation(
    UniqueLiveRange ulvr,
    size_t size,
    std::vector<MemAllocation>& ordered_allocations,
    OffsetFinder findOffset);
} // namespace jit
} // namespace torch