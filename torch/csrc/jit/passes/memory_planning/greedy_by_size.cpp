#include <torch/csrc/jit/passes/memory_planning/greedy_by_size.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_util.h>

namespace torch {
namespace jit {

std::vector<MemAllocation> greedyBySize(
    std::unordered_map<LiveRange, int64_t, live_range_hash>
        managed_live_ranges) {
  // sort tensor usage records in non-increasing order of size
  auto cmp =
      live_range_start_cmp();
  std::vector<LiveRange> sorted_size_live_ranges;
  std::transform(
      managed_live_ranges.begin(),
      managed_live_ranges.end(),
      std::back_inserter(sorted_size_live_ranges),
      [](auto item) { return item.first; });
  std::sort(
      sorted_size_live_ranges.begin(),
      sorted_size_live_ranges.end(),
      [&managed_live_ranges, &cmp](LiveRange& lvr1, LiveRange& lvr2) {
        return managed_live_ranges[lvr1] == managed_live_ranges[lvr2]
            ? cmp(lvr1, lvr2)
            : managed_live_ranges[lvr1] > managed_live_ranges[lvr2];
      });

  std::vector<MemAllocation> ordered_allocations;

  for (auto& lvr : sorted_size_live_ranges) {
    makeAllocation(
        ordered_allocations,
        managed_live_ranges,
        lvr,
        findOffsetWithSmallestGap);
  }

  std::sort(
      ordered_allocations.begin(),
      ordered_allocations.end(),
      [&cmp](auto m1, auto m2) { return cmp(m1.lvr, m2.lvr); });
  return ordered_allocations;
}

std::vector<MemAllocation> greedyBySizeWithFirstGap(
    std::unordered_map<LiveRange, int64_t, live_range_hash>
        managed_live_ranges) {
  auto cmp = live_range_start_cmp();
  // sort tensor usage records in non-increasing order of size
  std::vector<LiveRange> sorted_size_live_ranges;
  std::transform(
      managed_live_ranges.begin(),
      managed_live_ranges.end(),
      std::back_inserter(sorted_size_live_ranges),
      [](auto item) { return item.first; });
  std::sort(
      sorted_size_live_ranges.begin(),
      sorted_size_live_ranges.end(),
      [&managed_live_ranges, &cmp](LiveRange& lvr1, LiveRange& lvr2) {
        return managed_live_ranges[lvr1] == managed_live_ranges[lvr2]
            ? cmp(lvr1, lvr2)
            : managed_live_ranges[lvr1] > managed_live_ranges[lvr2];
      });

  std::vector<MemAllocation> ordered_allocations;

  for (auto& lvr : sorted_size_live_ranges) {
    makeAllocation(
        ordered_allocations, managed_live_ranges, lvr, findFirstOffset);
  }

  std::sort(
      ordered_allocations.begin(),
      ordered_allocations.end(),
      [&cmp](auto m1, auto m2) { return cmp(m1.lvr, m2.lvr); });
  return ordered_allocations;
}

std::vector<MemAllocation> greedyBySizeAndLongestWithFirstGap(
    std::unordered_map<LiveRange, int64_t, live_range_hash>
        managed_live_ranges) {
  auto cmp = live_range_start_cmp();
  // sort tensor usage records in non-increasing order of size
  std::vector<LiveRange> sorted_length_then_size_live_ranges;
  std::transform(
      managed_live_ranges.begin(),
      managed_live_ranges.end(),
      std::back_inserter(sorted_length_then_size_live_ranges),
      [](auto item) { return item.first; });
  std::sort(
      sorted_length_then_size_live_ranges.begin(),
      sorted_length_then_size_live_ranges.end(),
      [&managed_live_ranges, &cmp](LiveRange& lvr1, LiveRange& lvr2) {
        auto len1 = lvr1.begin - lvr1.end;
        auto len2 = lvr2.begin - lvr2.end;
        return len1 == len2
            ? (managed_live_ranges[lvr2]
                   ? cmp(lvr1, lvr2)
                   : managed_live_ranges[lvr1] > managed_live_ranges[lvr2])
            : len1 > len2;
      });

  std::vector<MemAllocation> ordered_allocations;

  for (auto& lvr : sorted_length_then_size_live_ranges) {
    makeAllocation(
        ordered_allocations, managed_live_ranges, lvr, findFirstOffset);
  }

  std::sort(
      ordered_allocations.begin(),
      ordered_allocations.end(),
      [&cmp](auto m1, auto m2) { return cmp(m1.lvr, m2.lvr); });
  return ordered_allocations;
}

} // namespace jit
} // namespace torch