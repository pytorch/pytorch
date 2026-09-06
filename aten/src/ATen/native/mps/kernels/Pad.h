#pragma once
#include <c10/metal/common.h>

struct ConstantPadDenseParams {
  ::c10::metal::array<uint32_t, 3> input_sizes;
  ::c10::metal::array<uint32_t, 3> output_sizes;
  ::c10::metal::array<uint32_t, 3> left_pad;
};

template <typename idx_t = uint32_t>
struct ConstantPadNdParams {
  ::c10::metal::array<idx_t, ::c10::metal::max_ndim> output_sizes;
  ::c10::metal::array<idx_t, ::c10::metal::max_ndim> input_sizes;
  ::c10::metal::array<idx_t, ::c10::metal::max_ndim> input_strides;
  ::c10::metal::array<idx_t, ::c10::metal::max_ndim> output_strides;
  ::c10::metal::array<idx_t, ::c10::metal::max_ndim> left_pad;
  uint32_t ndim;
};
