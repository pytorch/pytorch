#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

template <typename T>
T complex_div(T a, T b) {
  auto denom = dot(b, b);
  return T(dot(a, b), a.y * b.x - a.x * b.y) / denom;
}

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

struct sin_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(precise::sin(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return precise::sin(static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    // sin(x+yi)=sin(x)cosh(y)+icos(x)sinh(y);
    auto sin_x = precise::sin(x.x);
    auto cosh_y = precise::cosh(x.y);
    auto cos_x = precise::cos(x.x);
    auto sinh_y = precise::sinh(x.y);
    return T(sin_x * cosh_y, cos_x * sinh_y);
  }
};

struct cos_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(precise::cos(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return precise::cos(static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    // cos(x+yi)=cos(x)cosh(y)-isin(x)sinh(y);
    auto sin_x = precise::sin(x.x);
    auto cosh_y = precise::cosh(x.y);
    auto cos_x = precise::cos(x.x);
    auto sinh_y = precise::sinh(x.y);
    return T(cos_x * cosh_y, -1 * sin_x * sinh_y);
  }
};

struct tan_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(precise::tan(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return precise::tan(static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    // tan(x+yi)=(tan(x) + itanh(y)) / (1 - i(tan(x) * tanh(y)))
    auto tan_x = precise::tan(x.x);
    auto tanh_y = precise::tanh(x.y);
    return complex_div(T(tan_x, tanh_y), T(1, -1 * tan_x * tanh_y));
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

struct exp2_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(::precise::pow(2, x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return ::precise::pow(2, static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    // based on https://mathworld.wolfram.com/ComplexExponentiation.html
    auto coef = ::precise::pow(4, x.x / 2);
    auto ln = ::precise::log(4);
    auto real = ::precise::cos(0.5 * x.y * ln);
    auto imag = ::precise::sin(0.5 * x.y * ln);
    return T(coef * real, coef * imag);
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

struct rsqrt_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(1 / ::precise::sqrt(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return 1 / ::precise::sqrt(static_cast<float>(x));
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
    auto denominator = (real_part * real_part) + (imag_part * imag_part);
    return T(real_part / denominator, -1 * imag_part / denominator);
  }
};

struct neg_functor {
  template <typename T>
  inline T operator()(const T x) {
    return T(-1 * x);
  }
};

struct bitwise_not_functor {
  template <typename T>
  inline enable_if_t<!is_same_v<T, bool> && is_scalar_integral_v<T>, T>
  operator()(const T x) {
    return ~x;
  }

  template <typename T>
  inline enable_if_t<is_same_v<T, bool>, T> operator()(const T x) {
    return !x;
  }
};

DEFINE_UNARY_FLOATING_FUNCTOR(erfinv);
DEFINE_UNARY_FLOATING_FUNCTOR(sinc);

REGISTER_UNARY_OP(neg, int, int);
REGISTER_UNARY_OP(neg, long, long);
REGISTER_UNARY_OP(neg, short, short);
REGISTER_UNARY_OP(neg, char, char);
REGISTER_UNARY_OP(neg, uchar, uchar);
REGISTER_UNARY_OP(neg, float, float);
REGISTER_UNARY_OP(neg, half, half);

REGISTER_UNARY_OP(bitwise_not, int, int);
REGISTER_UNARY_OP(bitwise_not, long, long);
REGISTER_UNARY_OP(bitwise_not, short, short);
REGISTER_UNARY_OP(bitwise_not, char, char);
REGISTER_UNARY_OP(bitwise_not, uchar, uchar);
REGISTER_UNARY_OP(bitwise_not, bool, bool);

#define INSTANTIATE_UNARY_KERNELS2(DTYPE0, DTYPE1) \
  REGISTER_UNARY_OP(erfinv, DTYPE1, DTYPE0);       \
  REGISTER_UNARY_OP(exp, DTYPE1, DTYPE0);          \
  REGISTER_UNARY_OP(exp2, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(sinc, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(sqrt, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(rsqrt, DTYPE1, DTYPE0);        \
  REGISTER_UNARY_OP(tanh, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(sin, DTYPE1, DTYPE0);          \
  REGISTER_UNARY_OP(cos, DTYPE1, DTYPE0);          \
  REGISTER_UNARY_OP(tan, DTYPE1, DTYPE0)

#if __METAL_VERSION__ >= 310
INSTANTIATE_UNARY_KERNELS2(bfloat, bfloat);
REGISTER_UNARY_OP(neg, bfloat, bfloat);
#endif
INSTANTIATE_UNARY_KERNELS2(half, half);
INSTANTIATE_UNARY_KERNELS2(float, float);
INSTANTIATE_UNARY_KERNELS2(float, bool);
INSTANTIATE_UNARY_KERNELS2(float, uchar);
INSTANTIATE_UNARY_KERNELS2(float, char);
INSTANTIATE_UNARY_KERNELS2(float, short);
INSTANTIATE_UNARY_KERNELS2(float, int);
INSTANTIATE_UNARY_KERNELS2(float, long);

#define INSTANTIATE_UNARY_KERNELS_VEC2(DTYPE)   \
  REGISTER_UNARY_OP(neg, DTYPE##2, DTYPE##2);   \
  REGISTER_UNARY_OP(exp, DTYPE##2, DTYPE##2);   \
  REGISTER_UNARY_OP(exp2, DTYPE##2, DTYPE##2);  \
  REGISTER_UNARY_OP(tanh, DTYPE##2, DTYPE##2);  \
  REGISTER_UNARY_OP(sqrt, DTYPE##2, DTYPE##2);  \
  REGISTER_UNARY_OP(rsqrt, DTYPE##2, DTYPE##2); \
                                                \
  REGISTER_UNARY_OP(sinc, DTYPE##2, DTYPE##2);  \
  REGISTER_UNARY_OP(sin, DTYPE##2, DTYPE##2);   \
  REGISTER_UNARY_OP(cos, DTYPE##2, DTYPE##2);   \
  REGISTER_UNARY_OP(tan, DTYPE##2, DTYPE##2)

INSTANTIATE_UNARY_KERNELS_VEC2(half);
INSTANTIATE_UNARY_KERNELS_VEC2(float);

template <typename T>
kernel void round_decimals_dense(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant long& ndigits [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = static_cast<T>(
      rint(exp10(float(ndigits)) * input[index]) * exp10(float(-ndigits)));
}

template <typename T>
kernel void round_decimals_strided(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant long* sizes [[buffer(2)]],
    constant long* input_strides [[buffer(3)]],
    constant long* output_strides [[buffer(4)]],
    constant uint& ndim [[buffer(5)]],
    constant long& ndigits [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
  int pos[max_ndim];
  pos_from_thread_index(int(index), pos, sizes, ndim);
  const auto input_offs = offset_from_coord(pos, input_strides, ndim);
  const auto output_offs = offset_from_coord(pos, output_strides, ndim);
  output[output_offs] = static_cast<T>(
      rint(exp10(float(ndigits)) * input[input_offs]) * exp10(float(-ndigits)));
}

#define INSTANTIATE_ROUND_DECIMALS(DTYPE)                                    \
  template                                                                   \
      [[host_name("round_decimals_dense_" #DTYPE "_" #DTYPE)]] kernel void   \
      round_decimals_dense(                                                  \
          device DTYPE* output [[buffer(0)]],                                \
          constant DTYPE* input [[buffer(1)]],                               \
          constant long& ndigits [[buffer(2)]],                              \
          uint index [[thread_position_in_grid]]);                           \
  template                                                                   \
      [[host_name("round_decimals_strided_" #DTYPE "_" #DTYPE)]] kernel void \
      round_decimals_strided(                                                \
          device DTYPE* output [[buffer(0)]],                                \
          constant DTYPE* input [[buffer(1)]],                               \
          constant long* sizes,                                              \
          constant long* input_strides,                                      \
          constant long* output_strides,                                     \
          constant uint& ndim,                                               \
          constant long& ndigits [[buffer(6)]],                              \
          uint index)

INSTANTIATE_ROUND_DECIMALS(float);
INSTANTIATE_ROUND_DECIMALS(half);
#if __METAL_VERSION__ >= 310
INSTANTIATE_ROUND_DECIMALS(bfloat);
#endif
