#include <torch/csrc/jit/passes/memory_planning/linear_scan.h>

namespace torch {
namespace jit {

void coalesce_avail(std::set<Region, region_size_cmp>& avail_regions) {
  std::vector<Region> offset_sorted_avail_regions(
      avail_regions.begin(), avail_regions.end());
  std::sort(
      offset_sorted_avail_regions.begin(),
      offset_sorted_avail_regions.end(),
      [](Region const& reg1, Region const& reg2) {
        return reg1.offset < reg2.offset;
      });

  std::vector<Region> coalesced;
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
std::unordered_map<LiveRange, Region, live_range_hash> linearScanHeuristic(
    std::unordered_map<LiveRange, uint64_t, live_range_hash>
        managed_live_ranges) {
  auto cmp = [](LiveRange& lvr1, LiveRange& lvr2) {
    return lvr1.begin < lvr2.begin;
  };
  std::vector<LiveRange> sorted_start_live_ranges;
  std::sort(
      sorted_start_live_ranges.begin(), sorted_start_live_ranges.end(), cmp);

  int curr_end_reg = 0;
  std::set<LiveRange, live_range_end_cmp> active;
  std::map<LiveRange, Region, live_range_start_cmp> alloced_regions;
  std::map<LiveRange, Region, live_range_start_cmp> currently_alloced_regions;
  std::set<Region, region_size_cmp> avail_regions;

  auto expire_old_intervals = [&](LiveRange& curr_range) {
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
    auto aligned_curr_size =
        MemoryPlanner::compute_aligned_tensor_size(curr_size);

    // check avail regions
    const Region* reg = nullptr;
    coalesce_avail(avail_regions);
    for (auto& avail_reg : avail_regions) {
      if (avail_reg.size >= aligned_curr_size) {
        reg = &avail_reg;
        break;
      }
    }

    if (reg != nullptr) {
      avail_regions.erase(*reg);
      currently_alloced_regions[curr_range] = {reg->offset, aligned_curr_size};
      // split region (potentially)
      if (reg->size - aligned_curr_size > 0) {
        avail_regions.insert(
            {reg->offset + aligned_curr_size, reg->size - aligned_curr_size});
      }
    } else {
      // if possible spill smallest farthest out alloc
      const LiveRange* swap_lvr = nullptr;
      if (!active.empty()) {
        for (auto lv = active.end(); *lv != curr_range; lv--) {
          auto alloced_reg = currently_alloced_regions[*lv];
          if (alloced_reg.size >= aligned_curr_size) {
            reg = &alloced_reg;
            swap_lvr = &*lv;
          } else {
            break;
          }
        }
      }

      // swap i.e. put new alloc in old spot and malloc old alloc
      if (reg != nullptr) {
        // put new alloc at base of old region
        currently_alloced_regions[curr_range] = {
            reg->offset, aligned_curr_size};
        // split region (potentially)
        if (reg->size - aligned_curr_size > 0) {
          avail_regions.insert(
              {reg->offset + aligned_curr_size, reg->size - aligned_curr_size});
        }
        auto spill_size = currently_alloced_regions[*swap_lvr].size;
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

  // last interval doesn't come back around to the top of the loop
  expire_old_intervals(sorted_start_live_ranges.back());

  std::unordered_map<LiveRange, Region, live_range_hash> allocations(
      alloced_regions.begin(), alloced_regions.end());
  return allocations;
}

} // namespace jit
} // namespace torch