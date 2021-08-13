#include <torch/csrc/jit/passes/memory_planning/greedy_by_size.h>

namespace torch {
namespace jit {

// std::map<const Value*, Region> greedyBySize(
std::unordered_map<const Value*, Region> greedyBySize(
    std::unordered_map<const Value*, uint64_t> managed_tensor_sizes,
    LiveRangesMap live_ranges) {
  // sort tensor usage records in non-increasing order of size
  auto cmp = [&](auto v1, auto v2) {
    return managed_tensor_sizes[v1] >= managed_tensor_sizes[v2];
  };
  std::map<const Value*, LiveRange, decltype(cmp)> sorted_size_live_ranges_map(
      cmp);
  for (const auto& item : live_ranges) {
    sorted_size_live_ranges_map.insert({item.first, item.second});
  }

  auto region_cmp = [&](auto v1, auto v2) {
    return v1.second.offset < v2.second.offset;
  };
  std::multiset<std::pair<const Value*, Region>, decltype(region_cmp)>
      ordered_allocations(region_cmp);

  for (const auto& item : sorted_size_live_ranges_map) {
    auto t_val = item.first;
    auto t_size =
        MemoryPlanner::compute_aligned_tensor_size(managed_tensor_sizes[t_val]);
    auto lvr = item.second;

    uint64_t prev_offset = 0;
    uint64_t smallest_gap = std::numeric_limits<uint64_t>::max();
    c10::optional<uint64_t> best_offset = c10::nullopt;

    for (const auto& item : ordered_allocations) {
      auto offset = item.second.offset;
      auto alloced_t_val = item.first;

      auto latest_begin = std::max(lvr.begin, live_ranges[alloced_t_val].begin);
      auto earliest_end = std::min(lvr.end, live_ranges[alloced_t_val].end);
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
    ordered_allocations.insert({t_val, {best_offset.value(), t_size}});
  }
  std::unordered_map<const Value*, Region> allocations;
  for (auto& item : ordered_allocations) {
    allocations[item.first] = item.second;
  }
  return allocations;
}
} // namespace jit
} // namespace torch