#pragma once

#include <torch/csrc/jit/passes/memory_planning.h>

namespace torch {
namespace jit {

struct _region_offset_cmp {
  inline bool operator()(
      const std::pair<LiveRange, Region>& v1,
      const std::pair<LiveRange, Region>& v2) const {
    return region_offset_cmp()(v1.second, v2.second);
  }
};

uint64_t findOffset(
    LiveRange live_range,
    uint64_t t_size,
    std::unordered_map<LiveRange, uint64_t, live_range_hash>
        managed_live_ranges,
    std::multiset<std::pair<LiveRange, Region>, _region_offset_cmp>
        ordered_allocations);
} // namespace jit
} // namespace torch