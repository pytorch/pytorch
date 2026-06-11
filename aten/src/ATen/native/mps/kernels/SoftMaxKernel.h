#pragma once
#include <c10/metal/common.h>

struct SoftmaxParams {
  uint32_t axis_size;
  uint32_t ndim;
  uint32_t stride_a;
  uint32_t stride_b;
  uint32_t stride_c;
  uint32_t num_chunks;
  ::c10::metal::array<uint32_t, 15> outer_sizes;
  ::c10::metal::array<uint32_t, 15> outer_strides_a;
  ::c10::metal::array<uint32_t, 15> outer_strides_b;
  ::c10::metal::array<uint32_t, 15> outer_strides_c;
};
