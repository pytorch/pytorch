#pragma once

#include <torch/csrc/jit/passes/memory_planning.h>

namespace torch {
namespace jit {

struct _region_offset_cmp {
  inline bool operator()(
      const std::pair<const Value*, Region>& v1,
      const std::pair<const Value*, Region>& v2) const {
    return region_offset_cmp()(v1.second, v2.second);
  }
};

uint64_t findOffset(
    LiveRange live_range,
    uint64_t t_size,
    std::unordered_map<const Value*, uint64_t> managed_tensor_sizes,
    LiveRangesMap live_ranges,
    std::multiset<std::pair<const Value*, Region>, _region_offset_cmp>
        ordered_allocations);
} // namespace jit
} // namespace torch