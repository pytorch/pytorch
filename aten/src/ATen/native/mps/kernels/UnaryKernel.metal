#include <c10/metal/special_math.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

template <typename T>
T complex_div(T a, T b) {
  auto denom = dot(b, b);
  return T(dot(a, b), a.y * b.x - a.x * b.y) / denom;
}

struct erfinv_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(erfinv(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return erfinv(static_cast<float>(x));
  }
};

struct exp_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(precise::exp(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return precise::exp(static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    return T(
        precise::exp(x.x) * precise::cos(x.y),
        precise::exp(x.x) * precise::sin(x.y));
  }
};

struct tanh_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(precise::tanh(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return precise::tanh(static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    // tanh(x+iy)=(tanh(x)+itan(y))/(1+itahnh(x)*tan(y));
    auto tanh_x = precise::tanh(x.x);
    auto tan_y = precise::tan(x.y);
    return complex_div(T(tanh_x, tan_y), T(1.0, tanh_x * tan_y));
  }
};

struct sinc_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(sinc(static_cast<float>(x)));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return sinc(static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    return T(sinc(static_cast<float2>(x)));
  }
};

struct sqrt_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(::precise::sqrt(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return ::precise::sqrt(static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    // modulus
    auto m = precise::sqrt(x.x * x.x + x.y * x.y);
    // real part: sqrt((m + a)/2)
    auto real_part = precise::sqrt((m + x.x) * .5);
    // imaginary part: sign(b) * sqrt((m - a)/2)
    auto imag_part = copysign(
        static_cast<decltype(x.y)>(precise::sqrt((m - x.x) * .5)), x.y);
    return T(real_part, imag_part);
  }
};

template <typename T, typename F>
kernel void unary_dense(
    device result_of<F, T>* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  F f;
  output[index] = f(input[index]);
}

#define REGISTER_UNARY_OP(NAME, DTYPE0, DTYPE1)                           \
  static_assert(                                                          \
      is_same_v<DTYPE1, result_of<NAME##_functor, DTYPE0>>,               \
      "Output dtype mismatch for unary op " #NAME " and input " #DTYPE0); \
  template [[host_name(#NAME "_" #DTYPE1 "_" #DTYPE0)]] kernel void       \
  unary_dense<DTYPE0, NAME##_functor>(                                    \
      device result_of<NAME##_functor, DTYPE0> * output,                  \
      constant DTYPE0 * input,                                            \
      uint tid)

template <typename T0>
kernel void exp_complex_kernel(
    device vec2type_t<T0>* output [[buffer(0)]],
    constant vec2type_t<T0>* input [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  exp_functor f;
  output[index] = f(input[index]);
}

template <typename T0>
kernel void sqrt_complex_kernel(
    device vec2type_t<T0>* output [[buffer(0)]],
    constant vec2type_t<T0>* input [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  sqrt_functor f;
  output[index] = f(input[index]);
}

template <typename T0>
kernel void tanh_complex_kernel(
    device vec2type_t<T0>* output [[buffer(0)]],
    constant vec2type_t<T0>* input [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  tanh_functor f;
  output[index] = f(input[index]);
}

template <typename T0>
kernel void sinc_complex_kernel(
    device vec2type_t<T0>* output [[buffer(0)]],
    constant vec2type_t<T0>* input [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = vec2type_t<T0>(sinc(float2(input[index])));
}

#define INSTANTIATE_UNARY_KERNELS2(DTYPE0, DTYPE1) \
  REGISTER_UNARY_OP(erfinv, DTYPE1, DTYPE0);       \
  REGISTER_UNARY_OP(exp, DTYPE1, DTYPE0);          \
  REGISTER_UNARY_OP(sinc, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(sqrt, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(tanh, DTYPE1, DTYPE0)

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

#define INSTANTIATE_UNARY_KERNELS_VEC2(DTYPE)  \
  REGISTER_UNARY_OP(exp, DTYPE##2, DTYPE##2);  \
  REGISTER_UNARY_OP(tanh, DTYPE##2, DTYPE##2); \
  REGISTER_UNARY_OP(sqrt, DTYPE##2, DTYPE##2); \
  REGISTER_UNARY_OP(sinc, DTYPE##2, DTYPE##2)

INSTANTIATE_UNARY_KERNELS_VEC2(half);
INSTANTIATE_UNARY_KERNELS_VEC2(float);

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
