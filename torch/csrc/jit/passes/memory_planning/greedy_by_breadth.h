#pragma once

#include <torch/csrc/jit/passes/memory_planning.h>

namespace torch {
namespace jit {

std::vector<MemAllocation> greedyByOperatorBreadth(
    const LivenessMap& liveness_map,
    const FastMap<const Value*, std::pair<UniqueLiveRange, size_t>>& managed_values);

} // namespace jit
} // namespace torch