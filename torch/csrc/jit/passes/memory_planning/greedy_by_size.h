#pragma once

#include <torch/csrc/jit/passes/memory_planning.h>

namespace torch {
namespace jit {

std::vector<MemAllocation> greedyBySize(
    std::unordered_map<LiveRange, int64_t, live_range_hash>
        managed_live_ranges);

std::vector<MemAllocation> greedyBySizeWithFirstGap(
    std::unordered_map<LiveRange, int64_t, live_range_hash>
    managed_live_ranges);

std::vector<MemAllocation> greedyBySizeAndLongestWithFirstGap(
    std::unordered_map<LiveRange, int64_t, live_range_hash>
    managed_live_ranges);

} // namespace jit
} // namespace torch