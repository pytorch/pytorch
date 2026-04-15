#pragma once
#include <c10/metal/common.h>

template <unsigned N = 5, typename idx_type_t = int32_t>
struct GridSamplerParams {
  int32_t sampler_dims;
  ::c10::metal::array<idx_type_t, N> output_sizes;
  ::c10::metal::array<idx_type_t, N> output_strides;
  ::c10::metal::array<idx_type_t, N> input_sizes;
  ::c10::metal::array<idx_type_t, N> input_strides;
  ::c10::metal::array<idx_type_t, N> grid_sizes;
  ::c10::metal::array<idx_type_t, N> grid_strides;
  bool align_corners;
};

template <unsigned N = 5, typename idx_type_t = int32_t>
struct GridSamplerBackwardParams {
  GridSamplerParams<N, idx_type_t> forward;
  ::c10::metal::array<idx_type_t, N> grad_output_strides;
  ::c10::metal::array<idx_type_t, N> grad_input_strides;
  idx_type_t grad_grid_sW;
  int32_t padding_mode;
};

struct GridSampler3DBackwardParams {
  int32_t interpolation_mode;
  int32_t padding_mode;
  bool align_corners;
  bool compute_grad_input;
  bool compute_grad_grid;
  ::c10::metal::array<int32_t, 5> input_sizes;
  ::c10::metal::array<int32_t, 5> output_sizes;
  ::c10::metal::array<int32_t, 5> input_strides;
  ::c10::metal::array<int32_t, 5> grad_input_strides;
  ::c10::metal::array<int32_t, 5> grad_grid_strides;
  ::c10::metal::array<int32_t, 5> grid_strides;
  ::c10::metal::array<int32_t, 5> grad_output_strides;
};
