#pragma once

#ifndef __METAL__
#include <array>
#define _ARRAY_NS std
#else
#include <metal_array>
#define _ARRAY_NS metal
#endif

// N is the maximum allowed number of dimensions in the input and outputs. The
// maximum allowed pooling dimensions is N-2, because the input may have up to 2
// leading dimensions that are not pooled. To support up to 3-D pooling, N=5 is
// the default.
template <unsigned N = 5>
struct PoolingParams {
  int32_t dims;
  int32_t pooling_dims;
  _ARRAY_NS::array<int64_t, N> input_sizes;
  _ARRAY_NS::array<int64_t, N> input_strides;
  _ARRAY_NS::array<int64_t, N> output_sizes;
  _ARRAY_NS::array<int64_t, N> output_strides;
  _ARRAY_NS::array<int64_t, N> indices_sizes;
  _ARRAY_NS::array<int64_t, N> indices_strides;
  _ARRAY_NS::array<int64_t, N - 2> kernel_size;
  _ARRAY_NS::array<int64_t, N - 2> stride;
  _ARRAY_NS::array<int64_t, N - 2> padding;
  _ARRAY_NS::array<int64_t, N - 2> dilation;
};
