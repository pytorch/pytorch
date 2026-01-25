#pragma once
#include <c10/metal/common.h>

template <unsigned N = 4, typename idx_type_t = int32_t>
struct GridSamplerBackwardParams {
  ::c10::metal::array<idx_type_t, N> grad_output_sizes;
  ::c10::metal::array<idx_type_t, N> grad_output_strides;

  ::c10::metal::array<idx_type_t, N> input_sizes;
  ::c10::metal::array<idx_type_t, N> input_strides;

  ::c10::metal::array<idx_type_t, N> grid_sizes;
  ::c10::metal::array<idx_type_t, N> grid_strides;

  ::c10::metal::array<idx_type_t, N> grad_input_strides;
  ::c10::metal::array<idx_type_t, N> grad_grid_strides;

  GridSamplerInterpolation interpolation_mode;
  GridSamplerPadding padding_mode;
  bool align_corners;
  idx_type_t grad_input_memory_span;
  bool input_requires_grad;
};
