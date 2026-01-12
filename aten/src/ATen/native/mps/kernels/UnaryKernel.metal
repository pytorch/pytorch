#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

struct angle_functor {
  template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
  inline T operator()(const T x) {
    return T(atan2(x.y, x.x), 0);
  }
  template <typename T, enable_if_t<is_scalar_floating_point_v<T>, bool> = true>
  inline T operator()(const T x) {
    return T(isnan(x) ? x : x < 0 ? M_PI_F : 0.0);
  }
  template <typename T, enable_if_t<is_scalar_integral_v<T>, bool> = true>
  inline float operator()(const T x) {
    return x < 0 ? M_PI_F : 0.0;
  }
};

// Implement exp wrapper for both real and complex types
template <typename T, enable_if_t<is_scalar_floating_point_v<T>, bool> = true>
inline T exp_(const T x) {
  return T(precise::exp(x));
}

template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
inline T exp_(const T x) {
  return T(
      precise::exp(x.x) * precise::cos(x.y),
      precise::exp(x.x) * precise::sin(x.y));
}

struct exp_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T x) {
    return exp_(x);
  }
  template <typename T, enable_if_t<is_scalar_integral_v<T>, bool> = true>
  inline float operator()(const T x) {
    return exp_(static_cast<float>(x));
  }
};

struct expm1_functor {
  template <typename T, enable_if_t<is_scalar_floating_point_v<T>, bool> = true>
  inline T operator()(const T x) {
    if (::metal::fabs(x) < 1e-5f) {
      return static_cast<T>(c10::metal::expm1f(static_cast<float>(x)));
    } else {
      return static_cast<T>(exp_(static_cast<float>(x)) - 1.0f);
    }
  }
  template <typename T, enable_if_t<is_scalar_integral_v<T>, bool> = true>
  inline float operator()(const T x) {
    return exp_(static_cast<float>(x)) - 1;
  }
  template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
  inline T operator()(const T x) {
    if (::precise::sqrt(dot(x, x)) < 1e-2) {
      return T(
          c10::metal::expm1f(x.x + ::precise::log(precise::cos(x.y))),
          exp_(x.x) * precise::sin(x.y));
    } else {
      return exp_(x) - T(1.0f, 0.0f);
    }
  }
};

struct sigmoid_functor {
  template <typename T, enable_if_t<is_scalar_floating_point_v<T>, bool> = true>
  inline T operator()(const T x) {
    return T(1.0f / (1.0f + exp_(-static_cast<float>(x))));
  }
  template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
  inline T operator()(const T x) {
    return c10::metal::div(T(1, 0), (T(1, 0) + exp_(-x)));
  }
  template <typename T, enable_if_t<is_scalar_integral_v<T>, bool> = true>
  inline float operator()(const T x) {
    return 1.0f / (1.0f + exp_(-static_cast<float>(x)));
  }
};

struct abs_functor {
  template <typename T, enable_if_t<!is_complex_v<T>, bool> = true>
  inline T operator()(const T x) {
    return static_cast<T>(precise::abs(x));
  }
  template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
  inline T operator()(const T x) {
    return T(::precise::sqrt(dot(x, x)), 0);
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
    return div(T(tan_x, tanh_y), T(1, -1 * tan_x * tanh_y));
  }
};

struct sinh_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(precise::sinh(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return precise::sinh(static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    // sinh(x) = (e^x - e^(-x)) / 2
    auto exp_1 =
        T(precise::exp(x.x) * precise::cos(x.y),
          precise::exp(x.x) * precise::sin(x.y));
    auto exp_2 =
        T(precise::exp(-x.x) * precise::cos(-x.y),
          precise::exp(-x.x) * precise::sin(-x.y));
    return div(exp_1 - exp_2, T(2, 0));
  }
};

struct cosh_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(precise::cosh(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return precise::cosh(static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    // cosh(x+iy)=(e^x + e^(-x)) / 2
    auto exp_1 =
        T(precise::exp(x.x) * precise::cos(x.y),
          precise::exp(x.x) * precise::sin(x.y));
    auto exp_2 =
        T(precise::exp(-x.x) * precise::cos(-x.y),
          precise::exp(-x.x) * precise::sin(-x.y));
    return div(exp_1 + exp_2, T(2, 0));
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
    return div(T(tanh_x, tan_y), T(1.0, tanh_x * tan_y));
  }
};

struct asin_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(precise::asin(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return precise::asin(static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    // asin(z) = atan(z/sqrt(1-z^2)) if z != ±1
    if (x.x == 1 && x.y == 0)
      return T(M_PI_F / 2, 0);
    else if (x.x == -1 && x.y == 0)
      return T(M_PI_F / -2, 0);
    auto sqrt_val = T(1, 0) - c10::metal::mul(x, x);
    // calculate sqrt
    // modulus
    auto m = precise::sqrt(sqrt_val.x * sqrt_val.x + sqrt_val.y * sqrt_val.y);
    // real part: sqrt((m + a)/2)
    auto real_part = precise::sqrt((m + sqrt_val.x) * .5);
    // imaginary part: sign(b) * sqrt((m - a)/2)
    auto imag_part = copysign(
        static_cast<decltype(x.y)>(precise::sqrt((m - sqrt_val.x) * .5)),
        sqrt_val.y);
    auto atan_val = div(x, T(real_part, imag_part));
    // calculate atan (see atan_functor)
    auto coef = div(T(1, 0), T(0, 2));
    auto log_arg =
        div(T(-1 * atan_val.x, 1 - atan_val.y), T(atan_val.x, 1 + atan_val.y));
    // Calculate log using method from log_functor
    auto magnitude =
        ::precise::sqrt(log_arg.x * log_arg.x + log_arg.y * log_arg.y);
    auto real = ::precise::log(magnitude);
    auto imag = (log_arg.x == 0 && log_arg.y == 0)
        ? 0
        : ::precise::atan2(log_arg.y, log_arg.x);
    // return coefficient * log value
    return c10::metal::mul(coef, T(real, imag));
  }
};

struct acos_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(precise::acos(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return precise::acos(static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    // acos(z) = pi/2 - asin(z) if z != ±1
    // calculate asin
    if (x.x == 1 && x.y == 0)
      return T(M_PI_F, 0);
    else if (x.x == -1 && x.y == 0)
      return T(-M_PI_F, 0);
    auto sqrt_val = T(1, 0) - c10::metal::mul(x, x);
    // calculate sqrt
    // modulus
    auto m = precise::sqrt(sqrt_val.x * sqrt_val.x + sqrt_val.y * sqrt_val.y);
    // real part: sqrt((m + a)/2)
    auto real_part = precise::sqrt((m + sqrt_val.x) * .5);
    // imaginary part: sign(b) * sqrt((m - a)/2)
    auto imag_part = copysign(
        static_cast<decltype(x.y)>(precise::sqrt((m - sqrt_val.x) * .5)),
        sqrt_val.y);
    auto atan_val = div(x, T(real_part, imag_part));
    // calculate atan (see atan_functor)
    auto coef = div(T(1, 0), T(0, 2));
    auto log_arg =
        div(T(-1 * atan_val.x, 1 - atan_val.y), T(atan_val.x, 1 + atan_val.y));
    // Calculate log using method from log_functor
    auto magnitude =
        ::precise::sqrt(log_arg.x * log_arg.x + log_arg.y * log_arg.y);
    auto real = ::precise::log(magnitude);
    auto imag = (log_arg.x == 0 && log_arg.y == 0)
        ? 0
        : ::precise::atan2(log_arg.y, log_arg.x);
    // return coefficient * log value
    return T(M_PI_F / 2, 0) - c10::metal::mul(coef, T(real, imag));
  }
};

struct atan_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(precise::atan(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return precise::atan(static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    // atan(z) = (1/2i)ln((i-z)/(i+z))
    auto coef = div(T(1, 0), T(0, 2));
    auto log_arg = div(T(-1 * x.x, 1 - x.y), T(x.x, 1 + x.y));
    // Calculate log using method from log_functor
    auto magnitude =
        ::precise::sqrt(log_arg.x * log_arg.x + log_arg.y * log_arg.y);
    auto real = ::precise::log(magnitude);
    auto imag = (log_arg.x == 0 && log_arg.y == 0)
        ? 0
        : ::precise::atan2(log_arg.y, log_arg.x);
    // return coefficient * log value
    return c10::metal::mul(coef, T(real, imag));
  }
};

// Bool specialization is need to workaround compiler crashes on MacOS-13
// Otherwise attempts to invoke will fail to create state object with error
// Error Domain=AGXMetal13_3 Code=3 "Compiler encountered an internal error"

struct log_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(::precise::log(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return ::precise::log(static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    // log(x+yi) = ln(sqrt(x^2 + y^2)) + iarctan(y/x)
    auto magnitude = ::precise::sqrt(x.x * x.x + x.y * x.y);
    auto real = ::precise::log(magnitude);
    auto imag = (x.x == 0 && x.y == 0) ? 0 : ::precise::atan2(x.y, x.x);
    return T(real, imag);
  }
  inline float operator()(const bool x) {
    return x ? 0 : -INFINITY;
  }
};

struct log10_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(::precise::log10(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return ::precise::log10(static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    // Base 10 complex log = ln(x+yi)/ln(10)
    auto magnitude = ::precise::sqrt(x.x * x.x + x.y * x.y);
    auto real = ::precise::log(magnitude);
    auto imag = (x.x == 0 && x.y == 0) ? 0 : ::precise::atan2(x.y, x.x);
    return div(T(real, imag), T(::precise::log(10), 0));
  }
  inline float operator()(const bool x) {
    return x ? 0 : -INFINITY;
  }
};

struct log1p_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(::c10::metal::log1p(float(x)));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return ::precise::log(1.0f + static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    // TODO: Implement proper log1p algorithm
    auto magnitude = ::precise::sqrt((1.0f + x.x) * (1.0f + x.x) + x.y * x.y);
    auto real = ::precise::log(magnitude);
    auto imag = (x.x == -1 && x.y == 0) ? 0 : ::precise::atan2(x.y, 1.0 + x.x);
    return T(real, imag);
  }
  inline float operator()(const bool x) {
    return x ? ::precise::log(2.0) : 0;
  }
};

struct log2_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(::precise::log2(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return ::precise::log2(static_cast<float>(x));
  }
  template <typename T>
  inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {
    // Base 10 complex log = ln(x+yi)/ln(2)
    auto magnitude = ::precise::sqrt(x.x * x.x + x.y * x.y);
    auto real = ::precise::log(magnitude);
    auto imag = (x.x == 0 && x.y == 0) ? 0 : ::precise::atan2(x.y, x.x);
    return div(T(real, imag), T(::precise::log(2), 0));
  }
  inline float operator()(const bool x) {
    return x ? 0 : -INFINITY;
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

struct sqr_functor {
  template <typename T>
  inline T operator()(const T x) {
    return c10::metal::mul(x, x);
  }
};

struct reciprocal_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return T(1.0 / float(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return 1.0 / float(x);
  }
  template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
  inline T operator()(const T x) {
    return c10::metal::div(T(1, 0), x);
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

struct round_decimals_functor {
  template <typename T>
  inline T operator()(const T x, const long ndigits) {
    return static_cast<T>(
        rint(exp10(float(ndigits)) * x) * exp10(float(-ndigits)));
  }
};

struct pow_scalar_functor {
  template <typename T, typename U>
  inline T operator()(const T x, const U y) {
    return static_cast<T>(::c10::metal::pow(x, y));
  }
};

struct round_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T x) {
    return static_cast<T>(rint(float(x)));
  }
  template <typename T, enable_if_t<is_scalar_integral_v<T>, bool> = true>
  inline T operator()(const T x) {
    return x;
  }
};

DEFINE_UNARY_FLOATING_FUNCTOR(erf);
DEFINE_UNARY_FLOATING_FUNCTOR(erfc);
DEFINE_UNARY_FLOATING_FUNCTOR(erfinv);
DEFINE_UNARY_FLOATING_FUNCTOR(sinc);

REGISTER_UNARY_OP(neg, int, int);
REGISTER_UNARY_OP(neg, long, long);
REGISTER_UNARY_OP(neg, short, short);
REGISTER_UNARY_OP(neg, char, char);
REGISTER_UNARY_OP(neg, uchar, uchar);
REGISTER_UNARY_OP(neg, float, float);
REGISTER_UNARY_OP(neg, half, half);
REGISTER_UNARY_OP(round, int, int);
REGISTER_UNARY_OP(round, long, long);
REGISTER_UNARY_OP(round, short, short);
REGISTER_UNARY_OP(round, char, char);
REGISTER_UNARY_OP(round, uchar, uchar);
REGISTER_UNARY_OP(round, float, float);
REGISTER_UNARY_OP(round, half, half);

REGISTER_UNARY_OP(sqr, char, char);
REGISTER_UNARY_OP(sqr, uchar, uchar);
REGISTER_UNARY_OP(sqr, short, short);
REGISTER_UNARY_OP(sqr, int, int);
REGISTER_UNARY_OP(sqr, long, long);
REGISTER_UNARY_OP(sqr, float, float);
REGISTER_UNARY_OP(sqr, bfloat, bfloat);
REGISTER_UNARY_OP(sqr, half, half);
REGISTER_UNARY_OP(sqr, float2, float2);
REGISTER_UNARY_OP(sqr, half2, half2);

REGISTER_UNARY_OP(bitwise_not, int, int);
REGISTER_UNARY_OP(bitwise_not, long, long);
REGISTER_UNARY_OP(bitwise_not, short, short);
REGISTER_UNARY_OP(bitwise_not, char, char);
REGISTER_UNARY_OP(bitwise_not, uchar, uchar);
REGISTER_UNARY_OP(bitwise_not, bool, bool);

REGISTER_UNARY_OP(abs, int, int);
REGISTER_UNARY_OP(abs, long, long);
REGISTER_UNARY_OP(abs, short, short);
REGISTER_UNARY_OP(abs, char, char);
REGISTER_UNARY_OP(abs, uchar, uchar);
REGISTER_UNARY_OP(abs, float, float);
REGISTER_UNARY_OP(abs, half, half);

#define INSTANTIATE_UNARY_KERNELS2(DTYPE0, DTYPE1) \
  REGISTER_UNARY_OP(angle, DTYPE1, DTYPE0);        \
  REGISTER_UNARY_OP(erf, DTYPE1, DTYPE0);          \
  REGISTER_UNARY_OP(erfc, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(erfinv, DTYPE1, DTYPE0);       \
  REGISTER_UNARY_OP(exp, DTYPE1, DTYPE0);          \
  REGISTER_UNARY_OP(expm1, DTYPE1, DTYPE0);        \
  REGISTER_UNARY_OP(sigmoid, DTYPE1, DTYPE0);      \
  REGISTER_UNARY_OP(exp2, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(log, DTYPE1, DTYPE0);          \
  REGISTER_UNARY_OP(log10, DTYPE1, DTYPE0);        \
  REGISTER_UNARY_OP(log1p, DTYPE1, DTYPE0);        \
  REGISTER_UNARY_OP(log2, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(sinc, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(sqrt, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(reciprocal, DTYPE1, DTYPE0);   \
  REGISTER_UNARY_OP(rsqrt, DTYPE1, DTYPE0);        \
  REGISTER_UNARY_OP(sinh, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(cosh, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(tanh, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(sin, DTYPE1, DTYPE0);          \
  REGISTER_UNARY_OP(cos, DTYPE1, DTYPE0);          \
  REGISTER_UNARY_OP(tan, DTYPE1, DTYPE0);          \
  REGISTER_UNARY_OP(asin, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(acos, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(atan, DTYPE1, DTYPE0)

INSTANTIATE_UNARY_KERNELS2(bfloat, bfloat);
REGISTER_UNARY_OP(neg, bfloat, bfloat);
REGISTER_UNARY_OP(round, bfloat, bfloat);
REGISTER_UNARY_OP(abs, bfloat, bfloat);
INSTANTIATE_UNARY_KERNELS2(half, half);
INSTANTIATE_UNARY_KERNELS2(float, float);
INSTANTIATE_UNARY_KERNELS2(float, bool);
INSTANTIATE_UNARY_KERNELS2(float, uchar);
INSTANTIATE_UNARY_KERNELS2(float, char);
INSTANTIATE_UNARY_KERNELS2(float, short);
INSTANTIATE_UNARY_KERNELS2(float, int);
INSTANTIATE_UNARY_KERNELS2(float, long);

#define INSTANTIATE_UNARY_KERNELS_VEC2(DTYPE)        \
  REGISTER_UNARY_OP(angle, DTYPE##2, DTYPE##2);      \
  REGISTER_UNARY_OP(neg, DTYPE##2, DTYPE##2);        \
  REGISTER_UNARY_OP(exp, DTYPE##2, DTYPE##2);        \
  REGISTER_UNARY_OP(expm1, DTYPE##2, DTYPE##2);      \
  REGISTER_UNARY_OP(sigmoid, DTYPE##2, DTYPE##2);    \
  REGISTER_UNARY_OP(abs, DTYPE##2, DTYPE##2);        \
  REGISTER_UNARY_OP(exp2, DTYPE##2, DTYPE##2);       \
  REGISTER_UNARY_OP(log, DTYPE##2, DTYPE##2);        \
  REGISTER_UNARY_OP(log10, DTYPE##2, DTYPE##2);      \
  REGISTER_UNARY_OP(log1p, DTYPE##2, DTYPE##2);      \
  REGISTER_UNARY_OP(log2, DTYPE##2, DTYPE##2);       \
  REGISTER_UNARY_OP(sinh, DTYPE##2, DTYPE##2);       \
  REGISTER_UNARY_OP(cosh, DTYPE##2, DTYPE##2);       \
  REGISTER_UNARY_OP(tanh, DTYPE##2, DTYPE##2);       \
  REGISTER_UNARY_OP(sqrt, DTYPE##2, DTYPE##2);       \
  REGISTER_UNARY_OP(reciprocal, DTYPE##2, DTYPE##2); \
  REGISTER_UNARY_OP(rsqrt, DTYPE##2, DTYPE##2);      \
                                                     \
  REGISTER_UNARY_OP(sinc, DTYPE##2, DTYPE##2);       \
  REGISTER_UNARY_OP(sin, DTYPE##2, DTYPE##2);        \
  REGISTER_UNARY_OP(cos, DTYPE##2, DTYPE##2);        \
  REGISTER_UNARY_OP(tan, DTYPE##2, DTYPE##2);        \
  REGISTER_UNARY_OP(asin, DTYPE##2, DTYPE##2);       \
  REGISTER_UNARY_OP(acos, DTYPE##2, DTYPE##2);       \
  REGISTER_UNARY_OP(atan, DTYPE##2, DTYPE##2)

INSTANTIATE_UNARY_KERNELS_VEC2(half);
INSTANTIATE_UNARY_KERNELS_VEC2(float);

REGISTER_UNARY_ALPHA_OP(round_decimals, float, long, float);
REGISTER_UNARY_ALPHA_OP(round_decimals, half, long, half);
REGISTER_UNARY_ALPHA_OP(round_decimals, bfloat, long, bfloat);

REGISTER_UNARY_ALPHA_OP(pow_scalar, float2, float2, float2);
REGISTER_UNARY_ALPHA_OP(pow_scalar, half2, float2, half2);
REGISTER_UNARY_ALPHA_OP(pow_scalar, float, float, float);
REGISTER_UNARY_ALPHA_OP(pow_scalar, half, float, half);
REGISTER_UNARY_ALPHA_OP(pow_scalar, bfloat, float, bfloat);
REGISTER_UNARY_ALPHA_OP(pow_scalar, uchar, int, uchar);
REGISTER_UNARY_ALPHA_OP(pow_scalar, char, int, char);
REGISTER_UNARY_ALPHA_OP(pow_scalar, short, int, short);
REGISTER_UNARY_ALPHA_OP(pow_scalar, int, int, int);
REGISTER_UNARY_ALPHA_OP(pow_scalar, long, int, long);
