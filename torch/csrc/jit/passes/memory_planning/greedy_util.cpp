#include <torch/csrc/jit/passes/memory_planning/greedy_by_size.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_util.h>

namespace torch {
namespace jit {

int64_t findOffsetWithSmallestGap(
    LiveRange unalloced_lvr,
    int64_t tensor_size,
    std::vector<MemAllocation> ordered_allocations) {
  int64_t best_gap = std::numeric_limits<int64_t>::max();
  c10::optional<int64_t> best_offset = c10::nullopt;
  int64_t prev_offset = 0;

  for (const auto& alloc : ordered_allocations) {
    if (!intersectLiveRange(alloc.lvr, unalloced_lvr)) {
      continue;
    }

    // don't simplify this to gap = a - b because you'll get buffer overflow...
    if (alloc.reg.offset >= prev_offset) {
      auto gap = alloc.reg.offset - prev_offset;
      if (tensor_size <= gap && gap < best_gap) {
        best_gap = gap;
        best_offset = c10::optional<int64_t>(prev_offset);
      }
    }
    prev_offset = std::max(prev_offset, alloc.reg.offset + alloc.reg.size);
  }
  if (!best_offset.has_value()) {
    best_offset = c10::optional<int64_t>(prev_offset);
  }
  return best_offset.value();
}

int64_t findFirstOffset(
    LiveRange live_range,
    int64_t tensor_size,
    std::vector<MemAllocation> ordered_allocations) {
  c10::optional<int64_t> best_offset = c10::nullopt;
  int64_t prev_offset = 0;

  for (const auto& alloc : ordered_allocations) {
    if (!intersectLiveRange(alloc.lvr, live_range)) {
      continue;
    }

    // don't simplify this to gap = a - b because you'll get buffer overflow...
    if (alloc.reg.offset >= prev_offset) {
      auto gap = alloc.reg.offset - prev_offset;
      if (tensor_size <= gap) {
        best_offset = c10::optional<int64_t>(prev_offset);
        break;
      }
    }
    prev_offset = std::max(prev_offset, alloc.reg.offset + alloc.reg.size);
  }
  if (!best_offset.has_value()) {
    best_offset = c10::optional<int64_t>(prev_offset);
  }
  return best_offset.value();
}

void makeAllocation(
    std::vector<MemAllocation>& ordered_allocations,
    std::unordered_map<LiveRange, int64_t, live_range_hash>
        managed_live_ranges,
    LiveRange unalloced_lvr,
    OffsetFinder findOffset) {
  auto tensor_size = MemoryPlanner::computeAlignedTensorSize(
      managed_live_ranges[unalloced_lvr]);
  auto offset = findOffset(unalloced_lvr, tensor_size, ordered_allocations);
  auto it = ordered_allocations.begin();
  while (it != ordered_allocations.end() && it->reg.offset <= offset) {
    ++it;
  }
  ordered_allocations.insert(
      it, MemAllocation{unalloced_lvr, MemRegion{offset, tensor_size}});
}

} // namespace jit
} // namespace torch