#pragma once

#include <torch/csrc/jit/runtime/static/impl.h>
#include <torch/csrc/jit/passes/memory_planning.h>

namespace torch {
namespace jit {

std::vector<MemAllocation> linearScanHeuristic(
    std::unordered_map<LiveRange, int64_t, live_range_hash>
    managed_live_ranges);

} // namespace jit
} // namespace torch
