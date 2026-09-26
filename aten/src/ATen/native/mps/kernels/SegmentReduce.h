#pragma once

#include <c10/metal/common.h>

namespace c10::metal {

enum class SegmentReduction : unsigned { Max, Mean, Min, Sum, Prod };

struct SegmentReduceParams {
  uint64_t outer;
  uint64_t inner;
  uint64_t segments;
  uint64_t axis_size;
  float initial;
  unsigned has_initial;
};

} // namespace c10::metal
