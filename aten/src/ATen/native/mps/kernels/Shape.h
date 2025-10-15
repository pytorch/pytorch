#pragma once
#include <c10/metal/common.h>

template <unsigned N = c10::metal::max_ndim, typename idx_type_t = int64_t>
struct CatLargeSharedParams {
  int32_t ndim;
  int32_t cat_dim;
  ::c10::metal::array<idx_type_t, N> output_strides;
  ::c10::metal::array<idx_type_t, N> output_sizes;
};

template <unsigned N = c10::metal::max_ndim, typename idx_type_t = int64_t>
struct CatLargeInputParams {
  idx_type_t cat_dim_offset;
  idx_type_t input_element_offset;
  ::c10::metal::array<idx_type_t, N> input_strides;
  ::c10::metal::array<idx_type_t, N> input_sizes;
};
