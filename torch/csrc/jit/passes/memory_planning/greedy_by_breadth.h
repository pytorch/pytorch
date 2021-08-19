#pragma once

#include <torch/csrc/jit/passes/memory_planning.h>

namespace torch {
namespace jit {

std::unordered_map<LiveRange, Region, live_range_hash> greedyByOperatorBreadth(
    std::unordered_map<const Value*, uint64_t> managed_tensor_sizes,
    LiveRangesMap live_ranges,
    std::vector<const Node*> ops);

} // namespace jit
} // namespace torch