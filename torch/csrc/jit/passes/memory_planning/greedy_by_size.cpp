#include <torch/csrc/jit/passes/memory_planning/greedy_by_size.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_util.h>

namespace torch {
namespace jit {

std::unordered_map<LiveRange, Region, live_range_hash> greedyBySize(
    std::unordered_map<LiveRange, uint64_t, live_range_hash>
        managed_live_ranges) {
  // sort tensor usage records in non-increasing order of size
  auto cmp = [&managed_live_ranges](LiveRange& lvr1, LiveRange& lvr2) {
    return managed_live_ranges[lvr1] >= managed_live_ranges[lvr2];
  };
  std::vector<LiveRange> sorted_size_live_ranges;
  std::sort(
      sorted_size_live_ranges.begin(), sorted_size_live_ranges.end(), cmp);

  std::multiset<std::pair<LiveRange, Region>, _region_offset_cmp>
      ordered_allocations;

  for (auto& lvr : sorted_size_live_ranges) {
    auto t_size =
        MemoryPlanner::compute_aligned_tensor_size(managed_live_ranges[lvr]);
    auto best_offset =
        findOffset(lvr, t_size, managed_live_ranges, ordered_allocations);
    ordered_allocations.insert(
        std::make_pair(lvr, Region{best_offset, t_size}));
  }
  std::unordered_map<LiveRange, Region, live_range_hash> allocations;
  for (auto& item : ordered_allocations) {
    allocations[item.first] = item.second;
  }
  return allocations;
}
} // namespace jit
} // namespace torch