#pragma once

#include <torch/csrc/jit/passes/memory_planning.h>

namespace torch {
namespace jit {



std::vector<MemAllocation> greedyBySizeWithSmallestGap(
    const LivenessMap& liveness_map, const SortedLiveRangeMap<size_t>& live_ranges);


std::vector<MemAllocation> greedyBySizeWithFirstGap(
    const LivenessMap& liveness_map, const SortedLiveRangeMap<size_t>& live_ranges);


std::vector<MemAllocation> greedyByLongestAndSizeWithFirstGap(
    const LivenessMap& liveness_map, const SortedLiveRangeMap<size_t>& live_ranges);


std::vector<MemAllocation> greedyByLongestAndSizeWithSmallestGap(
    const LivenessMap& liveness_map, const SortedLiveRangeMap<size_t>& live_ranges);

} // namespace jit
} // namespace torch