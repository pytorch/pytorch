#pragma once

#include <torch/csrc/jit/passes/memory_planning.h>

namespace torch {
namespace jit {

std::vector<MemAllocation> greedyBySize(SortedLiveRangeMap<size_t> live_ranges);

std::vector<MemAllocation> greedyBySizeWithFirstGap(
    SortedLiveRangeMap<size_t> live_ranges);

std::vector<MemAllocation> greedyBySizeAndLongestWithFirstGap(
    SortedLiveRangeMap<size_t> live_ranges);

} // namespace jit
} // namespace torch