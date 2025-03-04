#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void renorm(
    constant T* norm [[buffer(0)]],
    device T* factor [[buffer(1)]],
    constant float& maxnorm [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  constexpr auto eps = static_cast<T>(1e-7);
  constexpr T one = 1;
  factor[index] = norm[index] > maxnorm
      ? static_cast<T>(maxnorm / (norm[index] + eps))
      : one;
}

#define REGISTER_RENORM_OP(DTYPE)                                     \
  template [[host_name("renorm_" #DTYPE)]] kernel void renorm<DTYPE>( \
      constant DTYPE * norm [[buffer(0)]],                            \
      device DTYPE * factor [[buffer(1)]],                            \
      constant float& maxnorm [[buffer(2)]],                          \
      uint index [[thread_position_in_grid]]);

REGISTER_RENORM_OP(float);
REGISTER_RENORM_OP(half);
#if __METAL_VERSION__ >= 310
REGISTER_RENORM_OP(bfloat);
#endif
