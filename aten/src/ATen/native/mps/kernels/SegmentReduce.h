#pragma once
#include <c10/metal/common.h>

// Mirrors at::native::ReductionType.
enum class SegmentReductionType {
  MAX = 0,
  MEAN = 1,
  MIN = 2,
  SUM = 3,
  PROD = 4
};

struct SegmentReduceParams {
  long outer_offset;
  long inner_offset;
  long segment_count;
  long data_stride_axis;
  long data_size_axis;
  long output_stride_axis;
  long output_size_axis;
  long offsets_stride_axis;
  long offsets_size_axis;
  float initial;
  bool has_initial;
  int reduction;
};
