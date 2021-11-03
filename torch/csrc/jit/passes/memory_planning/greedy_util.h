#pragma once

#include <torch/csrc/jit/passes/memory_planning.h>

namespace torch {
namespace jit {

enum GAP_PRIORITY { FIRST, SMALLEST };

size_t findGapOffset(
    UniqueLiveRange unalloced_ulvr,
    size_t size,
    std::unordered_map<const Value*, MemAllocation> ordered_allocations,
    const LivenessMap& liveness_map,
    GAP_PRIORITY gap_priority);

void makeAllocation(
    UniqueLiveRange ulvr,
    size_t size,
    std::unordered_map<const Value*, MemAllocation>& current_allocations,
    const LivenessMap& liveness_map,
    GAP_PRIORITY gap_priority = GAP_PRIORITY::SMALLEST);

std::vector<MemAllocation> orderAllocations(
    const std::unordered_map<const Value*, MemAllocation>&
        current_allocations);

} // namespace jit
} // namespace torch