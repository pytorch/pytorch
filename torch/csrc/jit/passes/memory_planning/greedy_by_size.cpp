#include <torch/csrc/jit/passes/memory_planning/greedy_by_size.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_util.h>

namespace torch {
namespace jit {

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

  std::multiset<std::pair<const Value*, Region>, _region_offset_cmp>
      ordered_allocations;

  for (const auto& item : sorted_size_live_ranges_map) {
    auto t_val = item.first;
    auto lvr = item.second;
    auto t_size =
        MemoryPlanner::compute_aligned_tensor_size(managed_tensor_sizes[t_val]);
    auto best_offset = findOffset(
        lvr, t_size, managed_tensor_sizes, live_ranges, ordered_allocations);
    ordered_allocations.insert({t_val, {best_offset, t_size}});
  }
  std::unordered_map<const Value*, Region> allocations;
  for (auto& item : ordered_allocations) {
    allocations[item.first] = item.second;
  }
  return allocations;
}
} // namespace jit
} // namespace torch