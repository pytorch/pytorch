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

// Parameters for fractional max pooling 2D
template <typename idx_type_t = int32_t>
struct FractionalMaxPool2dParams {
  idx_type_t numBatch;
  idx_type_t numPlanes;
  idx_type_t inputH;
  idx_type_t inputW;
  idx_type_t outputH;
  idx_type_t outputW;
  idx_type_t poolSizeH;
  idx_type_t poolSizeW;
};

// Parameters for fractional max pooling 3D
template <typename idx_type_t = int32_t>
struct FractionalMaxPool3dParams {
  idx_type_t numBatch;
  idx_type_t numPlanes;
  idx_type_t inputT;
  idx_type_t inputH;
  idx_type_t inputW;
  idx_type_t outputT;
  idx_type_t outputH;
  idx_type_t outputW;
  idx_type_t poolSizeT;
  idx_type_t poolSizeH;
  idx_type_t poolSizeW;
};

// Parameters for fractional max pooling backward 2D
template <typename idx_type_t = int32_t>
struct FractionalMaxPool2dBackwardParams {
  idx_type_t numBatch;
  idx_type_t numPlanes;
  idx_type_t inputH;
  idx_type_t inputW;
  idx_type_t outputH;
  idx_type_t outputW;
};

// Parameters for fractional max pooling backward 3D
template <typename idx_type_t = int32_t>
struct FractionalMaxPool3dBackwardParams {
  idx_type_t numBatch;
  idx_type_t numPlanes;
  idx_type_t inputT;
  idx_type_t inputH;
  idx_type_t inputW;
  idx_type_t outputT;
  idx_type_t outputH;
  idx_type_t outputW;
};
