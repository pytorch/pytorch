#pragma once
#include <c10/metal/common.h>

// Tile of outputs computed by each thread of max_pool2d_s1_nchw
C10_METAL_CONSTEXPR int32_t max_pool2d_s1_tile_cols = 8;
C10_METAL_CONSTEXPR int32_t max_pool2d_s1_tile_rows = 2;

// N is the maximum allowed number of dimensions in the input and outputs. The
// maximum allowed pooling dimensions is N-2, because the input may have up to 2
// leading dimensions that are not pooled. To support up to 3-D pooling, N=5 is
// the default.
// tid_offset is the first output element of a chunked dispatch, see
// dispatch_1d_chunks.
template <unsigned N = 5, typename idx_type_t = int32_t>
struct PoolingParams {
  int32_t dims;
  int32_t pooling_dims;
  int64_t tid_offset;
  ::c10::metal::array<idx_type_t, N> input_sizes;
  ::c10::metal::array<int64_t, N> input_strides;
  ::c10::metal::array<idx_type_t, N> output_sizes;
  ::c10::metal::array<int64_t, N> output_strides;
  ::c10::metal::array<idx_type_t, N> indices_sizes;
  ::c10::metal::array<int64_t, N> indices_strides;
  ::c10::metal::array<idx_type_t, N - 2> kernel_size;
  ::c10::metal::array<idx_type_t, N - 2> stride;
  ::c10::metal::array<idx_type_t, N - 2> padding;
  ::c10::metal::array<idx_type_t, N - 2> dilation;
  bool return_indices;
  bool adaptive;
};

template <unsigned N = 5, typename idx_type_t = int32_t>
struct AvgPoolingParams {
  int32_t dims;
  int32_t pooling_dims;
  int64_t tid_offset;
  ::c10::metal::array<idx_type_t, N> input_sizes;
  ::c10::metal::array<int64_t, N> input_strides;
  ::c10::metal::array<idx_type_t, N> output_sizes;
  ::c10::metal::array<int64_t, N> output_strides;
  ::c10::metal::array<idx_type_t, N - 2> kernel_size;
  ::c10::metal::array<idx_type_t, N - 2> stride;
  ::c10::metal::array<idx_type_t, N - 2> padding;
  int32_t divisor_override;
  bool count_include_pad;
  bool has_divisor_override;
  bool adaptive;
};

template <unsigned N = 5, typename idx_type_t = int32_t>
struct PoolingBackwardParams {
  int32_t dims;
  int32_t pooling_dims;
  int64_t tid_offset;
  ::c10::metal::array<idx_type_t, N> grad_input_sizes;
  ::c10::metal::array<int64_t, N> grad_input_strides;
  ::c10::metal::array<idx_type_t, N> grad_output_sizes;
  ::c10::metal::array<int64_t, N> grad_output_strides;
  ::c10::metal::array<int64_t, N> indices_strides;
};

template <unsigned N = 5, typename idx_type_t = int32_t>
struct MaxUnpoolingParams {
  int32_t dims;
  int32_t pooling_dims;
  int64_t tid_offset;
  ::c10::metal::array<idx_type_t, N> input_sizes;
  ::c10::metal::array<int64_t, N> input_strides;
  ::c10::metal::array<idx_type_t, N> output_sizes;
  ::c10::metal::array<int64_t, N> output_strides;
  ::c10::metal::array<int64_t, N> indices_strides;
};
