#pragma once

struct SegmentReduceParams {
  long segment_count;
  long outer_offset;
  long inner_offset;
  long data_size_axis;
  bool is_initial_set;
};
