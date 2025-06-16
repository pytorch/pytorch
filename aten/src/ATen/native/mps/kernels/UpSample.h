#ifndef __METAL__
#include <array>
using ulong = unsigned long;
using namespace std;
#else
#include <metal_array>
using namespace metal;
#endif

template <unsigned N = 5>
struct TensorShapes {
  array<ulong, N> sizes;
  array<ulong, N> strides;
};
