#pragma once

#include <c10/metal/common.h>

namespace c10::metal {

enum class SegmentReduction : unsigned { Max, Mean, Min, Sum, Prod };

// Prefer 256 to 512/1024 to limit idle lanes on short segments.
// Wider groups can help very long segments, but are not a universal win.
C10_METAL_CONSTEXPR unsigned segment_reduce_threads = 256;
C10_METAL_CONSTEXPR unsigned segment_reduce_partials =
    segment_reduce_threads / simdgroup_size;
static_assert(segment_reduce_threads % simdgroup_size == 0);
static_assert(
    segment_reduce_threads > 0 &&
    segment_reduce_threads <= max_threadgroup_size);
static_assert(segment_reduce_partials <= simdgroup_size);

struct SegmentReduceParams {
  uint64_t inner;
  uint64_t segments;
  uint64_t axis_size;
  float initial;
  bool has_initial;
};

} // namespace c10::metal
