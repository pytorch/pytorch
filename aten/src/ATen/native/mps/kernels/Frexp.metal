#include <metal_stdlib>
using namespace metal;

// frexp kernel - decomposes floating point numbers into mantissa and exponent
// mantissa = frexp(x, &exponent) where x = mantissa * 2^exponent
// mantissa is in range [0.5, 1.0) or 0 for x = 0

template <typename T>
kernel void frexp_kernel(
    device const T* input [[buffer(0)]],
    device T* mantissa [[buffer(1)]],
    device int* exponent [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  T x = input[index];
  int exp;
  T mant = frexp(x, &exp);
  mantissa[index] = mant;
  exponent[index] = exp;
}

#define REGISTER_FREXP_KERNEL(DTYPE)                                        \
  template [[host_name("frexp_kernel_" #DTYPE)]] kernel void frexp_kernel<DTYPE>( \
      device const DTYPE* input [[buffer(0)]],                              \
      device DTYPE* mantissa [[buffer(1)]],                                 \
      device int* exponent [[buffer(2)]],                                   \
      uint index [[thread_position_in_grid]])

REGISTER_FREXP_KERNEL(float);
REGISTER_FREXP_KERNEL(half);
// Note: bfloat16 may need special handling as Metal's frexp might not support it directly
