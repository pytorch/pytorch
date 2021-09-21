#include <torch/csrc/jit/passes/memory_planning/greedy_by_size.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_util.h>
#include <torch/csrc/jit/runtime/static/memory_planner.h>

namespace torch {
namespace jit {

size_t findOffsetWithSmallestGap(
    UniqueLiveRange unalloced_ulvr,
    size_t size,
    std::vector<MemAllocation> ordered_allocations) {
  size_t best_gap = std::numeric_limits<size_t>::max();
  c10::optional<size_t> best_offset = c10::nullopt;
  size_t prev_offset = 0;

  for (const auto& alloc : ordered_allocations) {
    if (!overlapLiveRange(alloc.ulvr, unalloced_ulvr)) {
      continue;
    }

    // don't simplify this to gap = a - b because you'll get buffer overflow...
    if (alloc.reg.offset >= prev_offset) {
      auto gap = alloc.reg.offset - prev_offset;
      if (size <= gap && gap < best_gap) {
        best_gap = gap;
        best_offset = c10::optional<size_t>(prev_offset);
      }
    }
    prev_offset = std::max(prev_offset, alloc.reg.offset + alloc.reg.size);
  }
  if (!best_offset.has_value()) {
    best_offset = c10::optional<size_t>(prev_offset);
  }
  return best_offset.value();
}

size_t findFirstOffset(
    UniqueLiveRange unalloced_ulvr,
    size_t size,
    std::vector<MemAllocation> ordered_allocations) {
  c10::optional<size_t> best_offset = c10::nullopt;
  size_t prev_offset = 0;

  for (const auto& alloc : ordered_allocations) {
    if (!overlapLiveRange(alloc.ulvr, unalloced_ulvr)) {
      continue;
    }

    // don't simplify this to gap = a - b because you'll get buffer overflow...
    if (alloc.reg.offset >= prev_offset) {
      auto gap = alloc.reg.offset - prev_offset;
      if (size <= gap) {
        best_offset = c10::optional<size_t>(prev_offset);
        break;
      }
    }
    prev_offset = std::max(prev_offset, alloc.reg.offset + alloc.reg.size);
  }
  if (!best_offset.has_value()) {
    best_offset = c10::optional<size_t>(prev_offset);
  }
  return best_offset.value();
}

MemAllocation makeAllocation(
    UniqueLiveRange ulvr,
    size_t size,
    std::vector<MemAllocation>& ordered_allocations,
    OffsetFinder findOffset) {
  auto aligned_size = MemoryPlanner::compute_aligned_tensor_size(size);
  auto offset = findOffset(ulvr, aligned_size, ordered_allocations);
  auto it = ordered_allocations.begin();
  while (it != ordered_allocations.end() && it->reg.offset <= offset) {
    ++it;
  }
  auto mem_alloc = MemAllocation{ulvr, MemRegion{offset, aligned_size}};
  ordered_allocations.insert(
      it, mem_alloc);
  return mem_alloc;
}

} // namespace jit
} // namespace torch