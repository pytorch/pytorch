#pragma once

template <typename T>
struct SegmentReduceParams {
  long segment_count;
  long outer_offset;
  long inner_offset;
  long data_size_axis;
  bool is_initial_set;
  T initial_value;
};
