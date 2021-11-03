#include <torch/csrc/jit/passes/memory_planning/greedy_by_size.h>
#include <torch/csrc/jit/passes/memory_planning/greedy_util.h>

namespace torch {
namespace jit {

bool lenCmp(
    std::pair<UniqueLiveRange, size_t> p1,
    std::pair<UniqueLiveRange, size_t> p2) {
  auto ulvr1 = p1.first;
  auto size1 = p1.second;
  auto ulvr2 = p2.first;
  auto size2 = p2.second;
  auto cmp = liveRangeStartCmp();

  auto len1 = ulvr1.lvr.end - ulvr1.lvr.begin;
  auto len2 = ulvr2.lvr.end - ulvr2.lvr.begin;
  return len1 == len2 ? (size1 == size2 ? cmp(ulvr1, ulvr2) : size1 > size2)
                      : len1 > len2;
}

// sort tensor usage records in non-increasing order of size (breaking ties by
// comparing live range starts)
bool sizeCmp(
    std::pair<UniqueLiveRange, size_t> p1,
    std::pair<UniqueLiveRange, size_t> p2) {
  auto ulvr1 = p1.first;
  auto size1 = p1.second;
  auto ulvr2 = p2.first;
  auto size2 = p2.second;
  auto cmp = liveRangeStartCmp();

  return size1 == size2 ? cmp(ulvr1, ulvr2) : size1 > size2;
}

using Cmp = bool(
    (std::pair<UniqueLiveRange, size_t> p1,
     std::pair<UniqueLiveRange, size_t> p2));

std::vector<MemAllocation> greedyBy(
    Cmp cmp,
    GAP_PRIORITY gap_priority,
    const LivenessMap& liveness_map,
    const SortedLiveRangeMap<size_t>& live_ranges) {
  std::vector<std::pair<UniqueLiveRange, size_t>> sorted_size_live_ranges(
      live_ranges.begin(), live_ranges.end());
  std::sort(
      sorted_size_live_ranges.begin(), sorted_size_live_ranges.end(), cmp);

  std::unordered_map<const Value*, MemAllocation> current_allocations;
  for (auto& item : sorted_size_live_ranges) {
    auto ulvr = item.first;
    auto size = item.second;
    makeAllocation(ulvr, size, current_allocations, liveness_map, gap_priority);
  }

  return orderAllocations(current_allocations);
}

std::vector<MemAllocation> greedyBySizeWithSmallestGap(
    const LivenessMap& liveness_map,
    const SortedLiveRangeMap<size_t>& live_ranges) {
  return greedyBy(sizeCmp, GAP_PRIORITY::SMALLEST, liveness_map, live_ranges);
}

std::vector<MemAllocation> greedyBySizeWithFirstGap(
    const LivenessMap& liveness_map,
    const SortedLiveRangeMap<size_t>& live_ranges) {
  return greedyBy(sizeCmp, GAP_PRIORITY::FIRST, liveness_map, live_ranges);
}

std::vector<MemAllocation> greedyByLongestAndSizeWithSmallestGap(
    const LivenessMap& liveness_map,
    const SortedLiveRangeMap<size_t>& live_ranges) {
  return greedyBy(lenCmp, GAP_PRIORITY::SMALLEST, liveness_map, live_ranges);
}

std::vector<MemAllocation> greedyByLongestAndSizeWithFirstGap(
    const LivenessMap& liveness_map,
    const SortedLiveRangeMap<size_t>& live_ranges) {
  return greedyBy(lenCmp, GAP_PRIORITY::FIRST, liveness_map, live_ranges);
}

} // namespace jit
} // namespace torch
