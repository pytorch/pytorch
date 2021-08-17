#pragma once

#include <torch/csrc/jit/passes/memory_planning.h>

namespace torch {
namespace jit {

std::unordered_map<LiveRange, Region, live_range_hash> linearScanHeuristic(
    std::unordered_map<LiveRange, uint64_t, live_range_hash>
        managed_live_ranges);

} // namespace jit
} // namespace torch
