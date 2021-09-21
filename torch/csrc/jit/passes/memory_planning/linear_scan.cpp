#include <torch/csrc/jit/passes/memory_planning/linear_scan.h>
#include <torch/csrc/jit/runtime/static/memory_planner.h>

namespace torch {
namespace jit {

// join regions that are adjacent and free
void coalesce_avail(std::multiset<MemRegion, regionSizeCmp>& avail_regions) {
  std::vector<MemRegion> coalesced;
  std::multiset<MemRegion, regionOffsetCmp> offset_sorted_avail_regions(
      avail_regions.begin(), avail_regions.end());
  for (auto& reg : offset_sorted_avail_regions) {
    if (!coalesced.empty() &&
        coalesced.back().offset + coalesced.back().size == reg.offset) {
      coalesced.back().size += reg.size;
    } else {
      coalesced.emplace_back(reg);
    }
  }
  avail_regions.clear();
  for (const auto& reg : coalesced) {
    avail_regions.insert(reg);
  }
};

// https://www.usenix.org/legacy/events/vee05/full_papers/p132-wimmer.pdf
// the idea basically register allocation but adapted to tensors; since tensors
// aren't fixed "width" like registers
std::vector<MemAllocation> linearScanHeuristic(
    SortedLiveRangeMap<size_t> live_ranges) {
  // sorted by right endpoint so we can find latest ending range quickly
  EndSortedLiveRangeMap active;
  std::multiset<MemRegion, regionSizeCmp> avail_regions;
  SortedLiveRangeMap<MemRegion> allocated_ranges;

  size_t curr_end_offset = 0;
  auto allocate_inactive_ranges = [&](UniqueLiveRange curr_range) {
    // copy because we're going to be erasing
    EndSortedLiveRangeMap temp_active(active);
    for (auto& item : temp_active) {
      // inactive means ranges don't intersect live ranges
      if (item.first.lvr.end < curr_range.lvr.begin) {
        auto inactive = item.first;
        auto reg = item.second;
        active.erase(inactive);
        // allocate inactive range
        allocated_ranges.insert({inactive, reg});
        avail_regions.insert(reg);
      }
    }
  };

  for (auto& item : live_ranges) {
    auto curr_range = item.first;
    auto curr_size = item.second;
    allocate_inactive_ranges(curr_range);
    coalesce_avail(avail_regions);

    auto aligned_curr_size = MemoryPlanner::compute_aligned_tensor_size(curr_size);

    // find the "right" region; in order of preference:
    // 1. tightest fit free region i.e. smallest i.e. first match since
    // avail_regions is sorted by size
    // 2. swap with latest ending active live range that is big enough (spilling
    // that alloc to the end of the current allocs)
    // 3. brand new alloc all the way at the end of the currently allocated
    // memory space

    auto candidate_reg = std::find_if(
        avail_regions.begin(),
        avail_regions.end(),
        [&aligned_curr_size](auto avail_reg) {
          return avail_reg.size >= aligned_curr_size;
        });

    if (candidate_reg != avail_regions.end()) {
      size_t candidate_offset = candidate_reg->offset;
      size_t candidate_size = candidate_reg->size;

      // note this erases at the iter position (i.e. not all matching regions,
      // of which there might be multiple)
      avail_regions.erase(candidate_reg);
      active[curr_range] = {candidate_offset, aligned_curr_size};
      // if candidate_reg is bigger than we need then we can
      // split it and keep that available space for other live ranges
      if (candidate_size - aligned_curr_size > 0) {
        avail_regions.insert(MemRegion{
            candidate_offset + aligned_curr_size,
            candidate_size - aligned_curr_size});
      }
      continue;
    }

    TORCH_INTERNAL_ASSERT(
        candidate_reg == avail_regions.end(), "inconsistent state");

    // no avail regions - look for lvr swap candidate in reverse order of
    // endpoint (i.e. latest ending)
    auto swap_candidate = std::find_if(
        active.rbegin(),
        active.rend(),
        [&curr_range, &aligned_curr_size](auto item) {
          return item.first.lvr.end > curr_range.lvr.end &&
              item.second.size >= aligned_curr_size;
        });

    if (swap_candidate != active.rend()) {
      auto candidate_reg = swap_candidate->second;
      size_t candidate_offset = candidate_reg.offset;
      size_t candidate_size = candidate_reg.size;
      active[curr_range] = {candidate_offset, aligned_curr_size};
      // split region (potentially)
      if (candidate_size > aligned_curr_size) {
        avail_regions.insert(MemRegion{
            candidate_offset + aligned_curr_size,
            candidate_size - aligned_curr_size});
      }

      // insert spilled lvr/region at the end
      active[swap_candidate->first] = {curr_end_offset, candidate_size};
      curr_end_offset += candidate_size;
      continue;
    }

    TORCH_INTERNAL_ASSERT(
        candidate_reg == avail_regions.end() && swap_candidate == active.rend(),
        "inconsistent state");

    // create new alloc
    active[curr_range] = {curr_end_offset, aligned_curr_size};
    curr_end_offset += aligned_curr_size;
  }

  // expire any remaining intervals
  allocate_inactive_ranges(
      {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()});

  std::vector<MemAllocation> allocations;
  allocations.reserve(allocated_ranges.size());
  for (const auto& item : allocated_ranges) {
    allocations.push_back({item.first, item.second});
  }
  return allocations;
}

} // namespace jit
} // namespace torch