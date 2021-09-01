#include <torch/csrc/jit/passes/memory_planning/linear_scan.h>

namespace torch {
namespace jit {

void coalesce_avail(std::set<MemRegion, region_size_cmp>& avail_regions) {
  std::vector<MemRegion> coalesced;
  std::set<MemRegion, region_offset_cmp> offset_sorted_avail_regions(
      avail_regions.begin(), avail_regions.end());
  for (auto& offset_sorted_avail_region : offset_sorted_avail_regions) {
    if (!coalesced.empty() &&
        coalesced.back().offset + coalesced.back().size ==
            offset_sorted_avail_region.offset) {
      coalesced.back().size += offset_sorted_avail_region.size;
    } else {
      coalesced.emplace_back(offset_sorted_avail_region);
    }
  }
  avail_regions.clear();
  for (const auto& reg : coalesced) {
    avail_regions.insert(reg);
  }
};

// https://www.usenix.org/legacy/events/vee05/full_papers/p132-wimmer.pdf
std::vector<MemAllocation> linearScanHeuristic(
    std::unordered_map<LiveRange, int64_t, live_range_hash>
        managed_live_ranges) {
  std::vector<LiveRange> sorted_start_live_ranges;
  std::transform(
      managed_live_ranges.begin(),
      managed_live_ranges.end(),
      std::back_inserter(sorted_start_live_ranges),
      [](auto item) { return item.first; });
  std::sort(
      sorted_start_live_ranges.begin(),
      sorted_start_live_ranges.end(),
      live_range_start_cmp());

  int64_t curr_end_reg = 0;

  std::set<LiveRange, live_range_end_cmp> active;
  std::map<LiveRange, MemRegion, live_range_start_cmp> alloced_regions;
  std::map<LiveRange, MemRegion, live_range_start_cmp>
      currently_alloced_regions;
  std::set<MemRegion, region_size_cmp> avail_regions;

  auto expire_old_intervals = [&](LiveRange curr_range) {
    for (auto& dead_range : sorted_start_live_ranges) {
      if (dead_range.end >= curr_range.begin) {
        break;
      }
      if (!active.count(dead_range)) {
        continue;
      }

      active.erase(dead_range);
      alloced_regions[dead_range] = currently_alloced_regions[dead_range];
      avail_regions.insert(currently_alloced_regions[dead_range]);
      currently_alloced_regions.erase(dead_range);
    }
  };

  for (auto& curr_range : sorted_start_live_ranges) {
    expire_old_intervals(curr_range);

    auto curr_size = managed_live_ranges[curr_range];
    auto aligned_curr_size = MemoryPlanner::computeAlignedTensorSize(curr_size);
    // check avail regions
    const MemRegion* reg = nullptr;
    coalesce_avail(avail_regions);
    for (auto& avail_reg : avail_regions) {
      if (avail_reg.size >= aligned_curr_size) {
        reg = &avail_reg;
        break;
      }
    }

    if (reg != nullptr) {
      int64_t swap_offset = reg->offset;
      int64_t swap_size = reg->size;

      avail_regions.erase(*reg);
      currently_alloced_regions[curr_range] = {swap_offset, aligned_curr_size};
      // split region (potentially)
      if (swap_size - aligned_curr_size > 0) {
        avail_regions.insert(MemRegion{
          swap_offset + aligned_curr_size, swap_size - aligned_curr_size});
      }
    } else {
      // if possible spill smallest farthest out alloc
      const LiveRange* swap_lvr = nullptr;
      if (!active.empty()) {
        for (auto lv = active.rbegin();  lv != active.rend(); lv++) {
          if (*lv == curr_range) break;
          if (currently_alloced_regions[*lv].size >= aligned_curr_size) {
            reg = &currently_alloced_regions[*lv];
            swap_lvr = &*lv;
          } else {
            break;
          }
        }
      }

      // swap i.e. put new alloc in old spot and malloc old alloc
      if (swap_lvr != nullptr) {
        TORCH_INTERNAL_ASSERT(reg != nullptr);
        // grab these now because *reg will be invalidated as soon
        // as we insert into currently_alloced_regions
        int64_t swap_offset = reg->offset;
        int64_t swap_size = reg->size;

        // put new alloc at base of old region
        currently_alloced_regions[curr_range] = {
            swap_offset, aligned_curr_size};
        // split region (potentially)
        if (swap_size - aligned_curr_size > 0) {
          avail_regions.insert(MemRegion{
              swap_offset + aligned_curr_size, swap_size - aligned_curr_size});
        }
        int64_t spill_size = currently_alloced_regions[*swap_lvr].size;
        currently_alloced_regions[*swap_lvr] = {curr_end_reg, spill_size};
        curr_end_reg += spill_size;
      } else {
        // create new alloc
        currently_alloced_regions[curr_range] = {
            curr_end_reg, aligned_curr_size};
        curr_end_reg += aligned_curr_size;
      }
    }

    active.insert(curr_range);
  }

  // expire any remaining intervals
  expire_old_intervals(
      {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()});

  std::vector<MemAllocation> allocations;
  allocations.reserve(alloced_regions.size());
  for (const auto& item : alloced_regions) {
    allocations.push_back({item.first, item.second});
  }
  return allocations;
}

} // namespace jit
} // namespace torch