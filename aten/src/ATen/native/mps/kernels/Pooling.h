#pragma once
#include <c10/metal/common.h>

// N is the maximum allowed number of dimensions in the input and outputs. The
// maximum allowed pooling dimensions is N-2, because the input may have up to 2
// leading dimensions that are not pooled. To support up to 3-D pooling, N=5 is
// the default.
template <unsigned N = 5>
struct PoolingParams {
  int32_t dims;
  int32_t pooling_dims;
  ::c10::metal::array<int64_t, N> input_sizes;
  ::c10::metal::array<int64_t, N> input_strides;
  ::c10::metal::array<int64_t, N> output_sizes;
  ::c10::metal::array<int64_t, N> output_strides;
  ::c10::metal::array<int64_t, N> indices_sizes;
  ::c10::metal::array<int64_t, N> indices_strides;
  ::c10::metal::array<int64_t, N - 2> kernel_size;
  ::c10::metal::array<int64_t, N - 2> stride;
  ::c10::metal::array<int64_t, N - 2> padding;
  ::c10::metal::array<int64_t, N - 2> dilation;
};

template <unsigned N = 5>
struct PoolingBackwardParams {
  int32_t dims;
  int32_t pooling_dims;
  ::c10::metal::array<int64_t, N> grad_input_sizes;
  ::c10::metal::array<int64_t, N> grad_input_strides;
  ::c10::metal::array<int64_t, N> grad_output_sizes;
  ::c10::metal::array<int64_t, N> grad_output_strides;
  ::c10::metal::array<int64_t, N> indices_strides;
};
