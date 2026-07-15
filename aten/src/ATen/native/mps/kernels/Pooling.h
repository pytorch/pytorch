#pragma once
#include <c10/metal/common.h>

// N is the maximum allowed number of dimensions in the input and outputs. The
// maximum allowed pooling dimensions is N-2, because the input may have up to 2
// leading dimensions that are not pooled. To support up to 3-D pooling, N=5 is
// the default.
template <unsigned N = 5, typename idx_type_t = int32_t>
struct PoolingParams {
  int32_t dims;
  int32_t pooling_dims;
  ::c10::metal::array<idx_type_t, N> input_sizes;
  ::c10::metal::array<idx_type_t, N> input_strides;
  ::c10::metal::array<idx_type_t, N> output_sizes;
  ::c10::metal::array<idx_type_t, N> output_strides;
  ::c10::metal::array<idx_type_t, N> indices_sizes;
  ::c10::metal::array<idx_type_t, N> indices_strides;
  ::c10::metal::array<idx_type_t, N - 2> kernel_size;
  ::c10::metal::array<idx_type_t, N - 2> stride;
  ::c10::metal::array<idx_type_t, N - 2> padding;
  ::c10::metal::array<idx_type_t, N - 2> dilation;
  bool return_indices;
};

template <unsigned N = 5, typename idx_type_t = int32_t>
struct AvgPoolingParams {
  int32_t dims;
  int32_t pooling_dims;
  ::c10::metal::array<idx_type_t, N> input_sizes;
  ::c10::metal::array<idx_type_t, N> input_strides;
  ::c10::metal::array<idx_type_t, N> output_sizes;
  ::c10::metal::array<idx_type_t, N> output_strides;
  ::c10::metal::array<idx_type_t, N - 2> kernel_size;
  ::c10::metal::array<idx_type_t, N - 2> stride;
  ::c10::metal::array<idx_type_t, N - 2> padding;
  bool count_include_pad;
  bool has_divisor_override;
  int32_t divisor_override;
};

template <unsigned N = 5, typename idx_type_t = int32_t>
struct PoolingBackwardParams {
  int32_t dims;
  int32_t pooling_dims;
  ::c10::metal::array<idx_type_t, N> grad_input_sizes;
  ::c10::metal::array<idx_type_t, N> grad_input_strides;
  ::c10::metal::array<idx_type_t, N> grad_output_sizes;
  ::c10::metal::array<idx_type_t, N> grad_output_strides;
  ::c10::metal::array<idx_type_t, N> indices_strides;
};

template <unsigned N = 5, typename idx_type_t = int32_t>
struct MaxUnpoolingParams {
  int32_t dims;
  int32_t pooling_dims;
  ::c10::metal::array<idx_type_t, N> input_sizes;
  ::c10::metal::array<idx_type_t, N> input_strides;
  ::c10::metal::array<idx_type_t, N> output_sizes;
  ::c10::metal::array<idx_type_t, N> output_strides;
  ::c10::metal::array<idx_type_t, N> indices_strides;
};

// Both sides are contiguous NCHW; the host copies through contiguous
// temporaries otherwise.
template <typename idx_type_t = int32_t>
struct AdaptiveMaxPoolParams {
  idx_type_t plane_count; // N * C
  idx_type_t input_h;
  idx_type_t input_w;
  idx_type_t output_h;
  idx_type_t output_w;
};
