#include <torch/csrc/jit/passes/memory_planning/greedy_by_size.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_util.h>

namespace torch {
namespace jit {

std::vector<MemAllocation> greedyBySizeWithSmallestGap(
    SortedLiveRangeMap<size_t> live_ranges) {
  // sort tensor usage records in non-increasing order of size
  std::vector<std::pair<UniqueLiveRange, size_t>> sorted_size_live_ranges(
      live_ranges.begin(), live_ranges.end());
  std::sort(
      sorted_size_live_ranges.begin(),
      sorted_size_live_ranges.end(),
      [](auto& p1, auto& p2) { return p1.second > p2.second; });

  std::vector<MemAllocation> ordered_allocations;

  for (auto& item : sorted_size_live_ranges) {
    auto ulvr = item.first;
    auto size = item.second;
    makeAllocation(ulvr, size, ordered_allocations, findOffsetWithSmallestGap);
  }

  auto cmp = liveRangeStartCmp();
  std::sort(
      ordered_allocations.begin(),
      ordered_allocations.end(),
      [&cmp](auto m1, auto m2) { return cmp(m1.ulvr, m2.ulvr); });
  return ordered_allocations;
}

std::vector<MemAllocation> greedyBySizeWithFirstGap(
    SortedLiveRangeMap<size_t> live_ranges) {
  // sort tensor usage records in non-increasing order of size
  std::vector<std::pair<UniqueLiveRange, size_t>> sorted_size_live_ranges(
      live_ranges.begin(), live_ranges.end());
  std::sort(
      sorted_size_live_ranges.begin(),
      sorted_size_live_ranges.end(),
      [](auto& p1, auto& p2) { return p1.second > p2.second; });

  std::vector<MemAllocation> ordered_allocations;

  for (auto& item : sorted_size_live_ranges) {
    auto ulvr = item.first;
    auto size = item.second;
    makeAllocation(ulvr, size, ordered_allocations, findFirstOffset);
  }

  auto cmp = liveRangeStartCmp();
  std::sort(
      ordered_allocations.begin(),
      ordered_allocations.end(),
      [&cmp](auto m1, auto m2) { return cmp(m1.ulvr, m2.ulvr); });
  return ordered_allocations;
}

std::vector<MemAllocation> greedyByLongestAndSizeWithFirstGap(
    SortedLiveRangeMap<size_t> live_ranges) {
  // sort tensor usage records in non-increasing order of size
  std::vector<std::pair<UniqueLiveRange, size_t>>
      sorted_length_then_size_live_ranges(
          live_ranges.begin(), live_ranges.end());
  auto cmp = liveRangeStartCmp();
  std::sort(
      sorted_length_then_size_live_ranges.begin(),
      sorted_length_then_size_live_ranges.end(),
      [&cmp](auto& p1, auto& p2) {
        auto len1 = p1.first.lvr.end - p1.first.lvr.begin;
        auto len2 = p2.first.lvr.end - p2.first.lvr.begin;
        return len1 == len2 ? (p1.second == p2.second ? cmp(p1.first, p2.first)
                                                      : p1.second > p2.second)
                            : len1 > len2;
      });

  std::vector<MemAllocation> ordered_allocations;

  for (auto& item : sorted_length_then_size_live_ranges) {
    auto ulvr = item.first;
    auto size = item.second;
    makeAllocation(ulvr, size, ordered_allocations, findFirstOffset);
  }

  std::sort(
      ordered_allocations.begin(),
      ordered_allocations.end(),
      [&cmp](auto m1, auto m2) { return cmp(m1.ulvr, m2.ulvr); });
  return ordered_allocations;
}

std::vector<MemAllocation> greedyByLongestAndSizeWithSmallestGap(
    SortedLiveRangeMap<size_t> live_ranges) {
  // sort tensor usage records in non-increasing order of size
  std::vector<std::pair<UniqueLiveRange, size_t>>
      sorted_length_then_size_live_ranges(
          live_ranges.begin(), live_ranges.end());
  auto cmp = liveRangeStartCmp();
  std::sort(
      sorted_length_then_size_live_ranges.begin(),
      sorted_length_then_size_live_ranges.end(),
      [&cmp](auto& p1, auto& p2) {
        auto len1 = p1.first.lvr.end - p1.first.lvr.begin;
        auto len2 = p2.first.lvr.end - p2.first.lvr.begin;
        return len1 == len2 ? (p1.second == p2.second ? cmp(p1.first, p2.first)
                                                      : p1.second > p2.second)
                            : len1 > len2;
      });

  std::vector<MemAllocation> ordered_allocations;

  for (auto& item : sorted_length_then_size_live_ranges) {
    auto ulvr = item.first;
    auto size = item.second;
    makeAllocation(ulvr, size, ordered_allocations, findOffsetWithSmallestGap);
  }

  std::sort(
      ordered_allocations.begin(),
      ordered_allocations.end(),
      [&cmp](auto m1, auto m2) { return cmp(m1.ulvr, m2.ulvr); });
  return ordered_allocations;
}

} // namespace jit
} // namespace torch