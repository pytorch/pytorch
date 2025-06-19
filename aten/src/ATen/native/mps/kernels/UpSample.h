#pragma once

#ifndef __METAL__
#include <array>
using ulong = unsigned long;
#define _ARRAY_NS std
#else
#include <metal_array>
#define _ARRAY_NS metal
#endif

template <unsigned N = 5>
struct UpsampleParams {
  _ARRAY_NS::array<ulong, N> input_strides;
  _ARRAY_NS::array<ulong, N> input_sizes;
  _ARRAY_NS::array<ulong, N> output_strides;
  _ARRAY_NS::array<ulong, N> output_sizes;
  _ARRAY_NS::array<float, N - 2> scales;
  bool align_corners;
};
