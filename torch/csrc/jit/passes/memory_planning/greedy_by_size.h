#pragma once

#include <torch/csrc/jit/passes/memory_planning.h>

namespace torch {
namespace jit {

std::unordered_map<const Value*, Region> greedyBySize(
    std::unordered_map<const Value*, uint64_t> managed_tensor_sizes,
    LiveRangesMap live_ranges);

} // namespace jit
} // namespace torch