#pragma once

#include <torch/csrc/jit/passes/memory_planning.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {

using EndSortedLiveRangeMap =
    std::map<UniqueLiveRange, MemRegion, liveRangeEndCmp>;

std::vector<MemAllocation> linearScanHeuristic(
    SortedLiveRangeMap<size_t> live_ranges);

} // namespace jit
} // namespace torch
