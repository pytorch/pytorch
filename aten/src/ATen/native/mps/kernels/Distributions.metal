#include <c10/metal/random.h>
#include <metal_stdlib>

using namespace metal;

template <typename T>
kernel void cauchy(
    device T* output [[buffer(0)]],
    constant float2& median_sigma [[buffer(1)]],
    constant long2& seed_base_offset [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  long seed = seed_base_offset.x;
  long base_offset = seed_base_offset.y;
  float median = median_sigma.x;
  float sigma = median_sigma.y;
  // Generate uniform random in (0, 1)
  float u = c10::metal::rand(seed, base_offset + tid);

  // Clamp to avoid tan(+- pi / 2) singularities
  constexpr float eps = 1.19209e-07f;
  u = clamp(u, eps, 1.0f - eps);

  // Cauchy inverse CDF: median + sigma * tan(pi * (u - 0.5))
  float result = median + sigma * tan(M_PI_F * (u - 0.5f));

  output[tid] = static_cast<T>(result);
}

#define REGISTER_CAUCHY_OP(DTYPE)                                     \
  template [[host_name("cauchy_" #DTYPE)]] kernel void cauchy<DTYPE>( \
      device DTYPE*, constant float2&, constant long2&, uint);

REGISTER_CAUCHY_OP(float);
REGISTER_CAUCHY_OP(half);
REGISTER_CAUCHY_OP(bfloat);
