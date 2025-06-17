#ifndef __METAL__
#include <array>
using ulong = unsigned long;
using namespace std;
#else
#include <metal_array>
using namespace metal;
#endif

template <unsigned N = 5>
struct UpsampleParams {
  array<ulong, N> input_strides;
  array<ulong, N> input_sizes;
  array<ulong, N> output_strides;
  array<ulong, N> output_sizes;
  array<float, N - 2> scales;
  bool align_corners;
};
