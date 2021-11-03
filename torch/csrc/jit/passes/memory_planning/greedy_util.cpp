#include <torch/csrc/jit/passes/memory_planning/greedy_by_size.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_util.h>
#include <torch/csrc/jit/runtime/static/memory_planner.h>

namespace torch {
namespace jit {

size_t findGapOffset(
    UniqueLiveRange unalloced_ulvr,
    size_t size,
    std::unordered_map<const Value*, MemAllocation> ordered_allocations,
    const LivenessMap& liveness_map,
    GAP_PRIORITY gap_priority) {
  size_t best_gap = std::numeric_limits<size_t>::max();
  c10::optional<size_t> best_offset = c10::nullopt;
  size_t prev_offset = 0;

  std::vector<const MemAllocation*> overlapping_allocations;
  for (auto& item : liveness_map.at(unalloced_ulvr.id)) {
    if (ordered_allocations.count(item)) {
      overlapping_allocations.emplace_back(&ordered_allocations.at(item));
    }
  }
  std::sort(
      overlapping_allocations.begin(),
      overlapping_allocations.end(),
      [](auto* m1, auto* m2) { return m1->reg.offset < m2->reg.offset; });

  for (const auto& alloc : overlapping_allocations) {
    // don't simplify this to gap = a - b because you'll get buffer overflow...
    if (alloc->reg.offset >= prev_offset) {
      auto gap = alloc->reg.offset - prev_offset;
      if (size <= gap && gap < best_gap) {
        best_offset = c10::optional<size_t>(prev_offset);
        if (gap_priority == GAP_PRIORITY::FIRST)
          break;
        best_gap = gap;
      }
    }
    prev_offset = std::max(prev_offset, alloc->reg.nextOffset());
  }
  if (!best_offset.has_value()) {
    best_offset = c10::optional<size_t>(prev_offset);
  }
  return best_offset.value();
}

void makeAllocation(
    UniqueLiveRange ulvr,
    size_t size,
    std::unordered_map<const Value*, MemAllocation>& current_allocations,
    const LivenessMap& liveness_map,
    GAP_PRIORITY gap_priority) {
  auto aligned_size = MemoryPlanner::compute_aligned_tensor_size(size);
  auto offset = findGapOffset(
      ulvr, aligned_size, current_allocations, liveness_map, gap_priority);
  auto mem_alloc = MemAllocation{ulvr, MemRegion{offset, aligned_size}};
  current_allocations.insert({ulvr.id, mem_alloc});
}

std::vector<MemAllocation> orderAllocations(
    const std::unordered_map<const Value*, MemAllocation>&
        current_allocations) {
  std::vector<MemAllocation> ordered_allocations;
  ordered_allocations.reserve(current_allocations.size());
  for (auto& item : current_allocations) {
    ordered_allocations.emplace_back(item.second);
  }

  auto final_order_cmp = liveRangeStartCmp();
  std::sort(
      ordered_allocations.begin(),
      ordered_allocations.end(),
      [&final_order_cmp](auto m1, auto m2) {
        return final_order_cmp(m1.ulvr, m2.ulvr);
      });

  return ordered_allocations;
}

} // namespace jit
} // namespace torch
