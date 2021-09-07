#pragma once

#include <torch/csrc/jit/passes/memory_planning.h>

namespace torch {
namespace jit {

std::vector<MemAllocation> greedyBySizeWithSmallestGap(
    SortedLiveRangeMap<size_t> live_ranges);

std::vector<MemAllocation> greedyBySizeWithFirstGap(
    SortedLiveRangeMap<size_t> live_ranges);

std::vector<MemAllocation> greedyByLongestAndSizeWithFirstGap(
    SortedLiveRangeMap<size_t> live_ranges);

std::vector<MemAllocation> greedyByLongestAndSizeWithSmallestGap(
    SortedLiveRangeMap<size_t> live_ranges);

} // namespace jit
} // namespace torch