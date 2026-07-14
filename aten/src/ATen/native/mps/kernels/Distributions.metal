#include <c10/metal/random.h>
#include <metal_stdlib>

using namespace metal;

constant constexpr float eps = 1.19209e-07f;

template <typename T>
kernel void cauchy(
    device T* output [[buffer(0)]],
    constant float2& params [[buffer(1)]],
    constant long2& seed_base_offset [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  // Generate uniform random in (0, 1)
  float u = c10::metal::rand(seed_base_offset.x, seed_base_offset.y + tid);
  // Clamp to avoid tan(+- pi / 2) singularities
  u = clamp(u, eps, 1.0f - eps);
  // Cauchy inverse CDF: median + sigma * tan(pi * (u - 0.5))
  float result =
      params.x + params.y * ::metal::precise::tan(M_PI_F * (u - 0.5f));
  output[tid] = static_cast<T>(result);
}

template <typename T>
kernel void log_normal(
    device T* output [[buffer(0)]],
    constant float2& params [[buffer(1)]],
    constant long2& seed_base_offset [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  long offset = seed_base_offset.y + 2 * tid;
  float u1 =
      clamp(c10::metal::rand(seed_base_offset.x, offset), eps, 1.0f - eps);
  float u2 = c10::metal::rand(seed_base_offset.x, offset + 1);
  float z = ::metal::precise::sqrt(-2.0f * ::metal::precise::log(u1)) *
      ::metal::precise::cos(2.0f * M_PI_F * u2);
  float result = ::metal::precise::exp(params.x + params.y * z);
  output[tid] = static_cast<T>(result);
}

template <typename T>
kernel void geometric(
    device T* output [[buffer(0)]],
    constant float2& params [[buffer(1)]],
    constant long2& seed_base_offset [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  float u = c10::metal::rand(seed_base_offset.x, seed_base_offset.y + tid);
  u = clamp(u, eps, 1.0f - eps);
  float result = ceil(::metal::precise::log(u) / params.x);
  output[tid] = static_cast<T>(result);
}

#define REGISTER_OP(NAME, DTYPE)                                    \
  template [[host_name(#NAME "_" #DTYPE)]] kernel void NAME<DTYPE>( \
      device DTYPE*, constant float2&, constant long2&, uint)

REGISTER_OP(cauchy, float);
REGISTER_OP(cauchy, half);
REGISTER_OP(cauchy, bfloat);

REGISTER_OP(log_normal, float);
REGISTER_OP(log_normal, half);
REGISTER_OP(log_normal, bfloat);

REGISTER_OP(geometric, float);
REGISTER_OP(geometric, half);
REGISTER_OP(geometric, bfloat);
REGISTER_OP(geometric, int);
REGISTER_OP(geometric, long);
REGISTER_OP(geometric, short);
REGISTER_OP(geometric, char);
REGISTER_OP(geometric, uchar);
