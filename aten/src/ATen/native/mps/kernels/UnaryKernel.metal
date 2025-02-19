#include <c10/metal/special_math.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

template <typename T0, typename T1>
kernel void erfinv_kernel(
    device T0* output [[buffer(0)]],
    constant T1* input [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = T0(erfinv(input[index]));
}

template <typename T0, typename T1>
kernel void exp_kernel(
    device T0* output [[buffer(0)]],
    constant T1* input [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = T0(precise::exp(input[index]));
}

template <typename T0>
kernel void exp_complex_kernel(
    device vec2type_t<T0>* output [[buffer(0)]],
    constant vec2type_t<T0>* input [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  output[index].x =
      T0(precise::exp(input[index].x) * precise::cos(input[index].y));
  output[index].y =
      T0(precise::exp(input[index].x) * precise::sin(input[index].y));
}

template <typename T0, typename T1>
kernel void tanh_kernel(
    device T0* output [[buffer(0)]],
    constant T1* input [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = T0(precise::tanh(input[index]));
}

#if __METAL_VERSION__ >= 310
bfloat dot(bfloat2 a, bfloat2 b) {
  return a.x * b.x + a.y * b.y;
}
#endif

short dot(short2 a, short2 b) {
  return a.x * b.x + a.y * b.y;
}

template <typename T>
T complex_div(T a, T b) {
  auto denom = dot(b, b);
  return T(dot(a, b), a.y * b.x - a.x * b.y) / denom;
}

template <typename T0>
kernel void tanh_complex_kernel(
    device vec2type_t<T0>* output [[buffer(0)]],
    constant vec2type_t<T0>* input [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  // tanh(x+iy)=(tanh(x)+itan(y))/(1+itahnh(x)*tan(y));
  auto tanh_x = T0(precise::tanh(input[index].x));
  auto tan_y = T0(precise::tan(input[index].y));
  output[index] = complex_div(
      vec2type_t<T0>(tanh_x, tan_y), vec2type_t<T0>(T0(1), tanh_x * tan_y));
}

#define INSTANTIATE_UNARY_KERNELS2(DTYPE0, DTYPE1)                             \
  template [[host_name("erfinv_" #DTYPE0 "_" #DTYPE1)]] kernel void            \
  erfinv_kernel(                                                               \
      device DTYPE0* output [[buffer(0)]],                                     \
      constant DTYPE1* input [[buffer(1)]],                                    \
      uint id [[thread_position_in_grid]]);                                    \
  template [[host_name("exp_" #DTYPE0 "_" #DTYPE1)]] kernel void exp_kernel(   \
      device DTYPE0* output [[buffer(0)]],                                     \
      constant DTYPE1* input [[buffer(1)]],                                    \
      uint id [[thread_position_in_grid]]);                                    \
  template [[host_name("tanh_" #DTYPE0 "_" #DTYPE1)]] kernel void tanh_kernel( \
      device DTYPE0* output [[buffer(0)]],                                     \
      constant DTYPE1* input [[buffer(1)]],                                    \
      uint id [[thread_position_in_grid]]);

#if __METAL_VERSION__ >= 310
INSTANTIATE_UNARY_KERNELS2(bfloat, bfloat);
#endif
INSTANTIATE_UNARY_KERNELS2(half, half);
INSTANTIATE_UNARY_KERNELS2(float, float);
INSTANTIATE_UNARY_KERNELS2(float, bool);
INSTANTIATE_UNARY_KERNELS2(float, uchar);
INSTANTIATE_UNARY_KERNELS2(float, char);
INSTANTIATE_UNARY_KERNELS2(float, short);
INSTANTIATE_UNARY_KERNELS2(float, int);
INSTANTIATE_UNARY_KERNELS2(float, long);

#define INSTANTIATE_UNARY_KERNELS_VEC2(DTYPE0, DTYPE1)                    \
  template [[host_name("exp_complex_" #DTYPE0 "_" #DTYPE1)]] kernel void  \
  exp_complex_kernel<DTYPE0>(                                             \
      device vec2type_t<DTYPE0> * output [[buffer(0)]],                   \
      constant vec2type_t<DTYPE0> * input [[buffer(1)]],                  \
      uint did [[thread_position_in_grid]]);                              \
  template [[host_name("tanh_complex_" #DTYPE0 "_" #DTYPE1)]] kernel void \
  tanh_complex_kernel<DTYPE0>(                                            \
      device vec2type_t<DTYPE0> * output [[buffer(0)]],                   \
      constant vec2type_t<DTYPE0> * input [[buffer(1)]],                  \
      uint did [[thread_position_in_grid]]);

INSTANTIATE_UNARY_KERNELS_VEC2(short, short);
INSTANTIATE_UNARY_KERNELS_VEC2(float, float);

template <typename T0, typename T1>
kernel void sinc_kernel(
    device T0* output [[buffer(0)]],
    constant T1* input [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = T0(sinc(static_cast<float>(input[index])));
}

template <typename T>
kernel void sinc_complex(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = T(sinc(float2(input[index])));
}

#define INSTANTIATE_SINC_KERNEL(DTYPE0, DTYPE1)                                \
  template [[host_name("sinc_" #DTYPE0 "_" #DTYPE1)]] kernel void sinc_kernel( \
      device DTYPE0* output [[buffer(0)]],                                     \
      constant DTYPE1* input [[buffer(1)]],                                    \
      uint id [[thread_position_in_grid]])

#define INSTANTIATE_SINC_COMPLEX_KERNEL(DTYPE)                          \
  template [[host_name("sinc_complex_" #DTYPE "_" #DTYPE)]] kernel void \
  sinc_complex(                                                         \
      device DTYPE##2 * output [[buffer(0)]],                           \
      constant DTYPE##2 * input [[buffer(1)]],                          \
      uint id [[thread_position_in_grid]])

#if __METAL_VERSION__ >= 310
INSTANTIATE_SINC_KERNEL(bfloat, bfloat);
#endif
INSTANTIATE_SINC_KERNEL(half, half);
INSTANTIATE_SINC_KERNEL(float, float);
INSTANTIATE_SINC_KERNEL(float, long);
INSTANTIATE_SINC_KERNEL(float, int);
INSTANTIATE_SINC_KERNEL(float, short);
INSTANTIATE_SINC_KERNEL(float, char);
INSTANTIATE_SINC_KERNEL(float, uchar);
INSTANTIATE_SINC_KERNEL(float, bool);
INSTANTIATE_SINC_COMPLEX_KERNEL(half);
INSTANTIATE_SINC_COMPLEX_KERNEL(float);

template <typename T>
kernel void round_decimals_kernel(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant long& ndigits [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = static_cast<T>(
      rint(exp10(float(ndigits)) * input[index]) * exp10(float(-ndigits)));
}

#define INSTANTIATE_ROUND_DECIMALS(DTYPE)                                 \
  template [[host_name("round_decimals_" #DTYPE "_" #DTYPE)]] kernel void \
  round_decimals_kernel(                                                  \
      device DTYPE* output [[buffer(0)]],                                 \
      constant DTYPE* input [[buffer(1)]],                                \
      constant long& ndigits [[buffer(2)]],                               \
      uint id [[thread_position_in_grid]])

INSTANTIATE_ROUND_DECIMALS(float);
INSTANTIATE_ROUND_DECIMALS(half);
#if __METAL_VERSION__ >= 310
INSTANTIATE_ROUND_DECIMALS(bfloat);
#endif
