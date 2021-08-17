#include <torch/csrc/jit/passes/memory_planning/greedy_by_size.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_util.h>

namespace torch {
namespace jit {

uint64_t findOffset(
    LiveRange live_range,
    uint64_t t_size,
    std::unordered_map<const Value*, uint64_t> managed_tensor_sizes,
    LiveRangesMap live_ranges,
    std::multiset<std::pair<const Value*, Region>, _region_offset_cmp>
        ordered_allocations) {
  uint64_t prev_offset = 0;
  uint64_t smallest_gap = std::numeric_limits<uint64_t>::max();
  c10::optional<uint64_t> best_offset = c10::nullopt;

  for (const auto& item : ordered_allocations) {
    auto offset = item.second.offset;
    auto alloced_t_val = item.first;

    auto latest_begin =
        std::max(live_range.begin, live_ranges[alloced_t_val].begin);
    auto earliest_end =
        std::min(live_range.end, live_ranges[alloced_t_val].end);
    if (latest_begin <= earliest_end) {
      auto gap = offset - prev_offset;
      if (gap >= t_size && gap < smallest_gap) {
        smallest_gap = gap;
        best_offset = c10::optional<uint64_t>(prev_offset);
      }
      auto alloced_t_size = MemoryPlanner::compute_aligned_tensor_size(
          managed_tensor_sizes[alloced_t_val]);
      prev_offset = std::max(prev_offset, offset + alloced_t_size);
    }
  }
  if (!best_offset.has_value()) {
    best_offset = c10::optional<uint64_t>(prev_offset);
  }
  return best_offset.value();
}

} // namespace jit
} // namespace torch