#include <ATen/native/mps/kernels/UnaryKernel.h>
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
  auto ex = precise::exp(x.x);
  // y == 0: avoid inf*0 / nan*0 = NaN in imag (matches C99 cexp).
  if (x.y == 0) {
    return T(ex, 0);
  }
  // Metal lacks a half sincos; do it in float (free for float complex, more
  // accurate for half complex).
  float c;
  float s = precise::sincos(static_cast<float>(x.y), c);
  return T(
      ex * static_cast<decltype(ex)>(c), ex * static_cast<decltype(ex)>(s));
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
    if (::metal::fabs(x) < 1e-1f) {
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
    // y == 0: same rationale as exp_ short-circuit above.
    if (x.y == 0) {
      return T(c10::metal::expm1f(x.x), 0);
    }
    if (::precise::sqrt(dot(x, x)) < 1e-2) {
      float c;
      float s = precise::sincos(static_cast<float>(x.y), c);
      using elem_t = decltype(x.x + x.x); // unwrapped value type
      return T(
          c10::metal::expm1f(x.x + ::precise::log(static_cast<elem_t>(c))),
          exp_(x.x) * static_cast<elem_t>(s));
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
  // precise:: has only float candidates, so bfloat converts unambiguously
  // (unqualified abs is ambiguous for bfloat: float vs half), and abs is
  // exact under the round trip. Integers must not take this path: the float
  // conversion rounds above 2^24.
  template <
      typename T,
      enable_if_t<!is_complex_v<T> && !is_scalar_integral_v<T>, bool> = true>
  inline T operator()(const T x) {
    return static_cast<T>(precise::abs(x));
  }
  template <typename T, enable_if_t<is_scalar_integral_v<T>, bool> = true>
  inline T operator()(const T x) {
    return ::metal::abs(x);
  }
  template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
  inline T operator()(const T x) {
    return T(c10::metal::hypot(x.x, x.y), 0);
  }
};

struct sign_functor {
  template <typename T>
  inline T operator()(const T x) {
    return T((T(0) < x) - (x < T(0)));
  }
};

struct sgn_functor {
  template <typename T>
  inline T operator()(const T x) {
    const auto z = float2(x);
    const auto m = c10::metal::hypot(z.x, z.y);
    return m == 0 ? T(0, 0) : T(z / m);
  }
};

struct signbit_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, bool> operator()(
      const T x) {
    return ::metal::signbit(float(x));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, bool> operator()(const T x) {
    return x < T(0);
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
    // sin(x+yi)=sin(x)cosh(y)+icos(x)sinh(y); float sincos covers half too.
    float cos_x;
    float sin_x = precise::sincos(static_cast<float>(x.x), cos_x);
    using elem_t = decltype(x.x + x.x);
    return T(
        static_cast<elem_t>(sin_x) * precise::cosh(x.y),
        static_cast<elem_t>(cos_x) * precise::sinh(x.y));
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
    // cos(x+yi)=cos(x)cosh(y)-isin(x)sinh(y); float sincos covers half too.
    float cos_x;
    float sin_x = precise::sincos(static_cast<float>(x.x), cos_x);
    using elem_t = decltype(x.x + x.x);
    return T(
        static_cast<elem_t>(cos_x) * precise::cosh(x.y),
        -static_cast<elem_t>(sin_x) * precise::sinh(x.y));
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
    // sinh(x) = (e^x - e^(-x)) / 2; delegate to exp_ to inherit y==0 fix.
    return div(exp_(x) - exp_(T(-x.x, -x.y)), T(2, 0));
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
    // cosh(x+iy) = (e^x + e^(-x)) / 2; delegate to exp_ to inherit y==0 fix.
    return div(exp_(x) + exp_(T(-x.x, -x.y)), T(2, 0));
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

// Complex inverse trigonometric/hyperbolic functions, ported from FreeBSD's
// catrigf.c, which implements Hull, Fairgrieve and Tang, "Implementing the
// complex arcsine and arccosine functions using exception handling" (1997).
// thrust (and therefore CUDA) uses the same algorithm. casin/cacosh/catan are
// derived from casinh/cacos/catanh.
namespace catrig {
constant constexpr float RECIP_EPSILON = 1 / FLT_EPSILON;
constant constexpr float A_CROSSOVER = 10;
constant constexpr float B_CROSSOVER = 0.6417;
constant constexpr float FOUR_SQRT_MIN = 0x1p-61;
constant constexpr float SQRT_6_EPSILON = 8.4572793338e-4;
constant constexpr float SQRT_3_EPSILON = 5.9801995673e-4;
constant constexpr float SQRT_MIN = 0x1p-63;

// (hypot(a, b) - b) / 2 computed without cancellation
inline float f(float a, float b, float hypot_a_b) {
  if (b < 0) {
    return (hypot_a_b - b) / 2;
  }
  if (b == 0) {
    return a / 2;
  }
  return a * a / (hypot_a_b + b) / 2;
}

struct hard_work_result {
  float rx; // acosh(A)
  bool B_is_usable;
  float B; // y / A
  float sqrt_A2my2; // when !B_is_usable, asin(B) == atan2(new_y, sqrt_A2my2)
  float new_y;
};

// For x, y >= 0 and A = (|x + i(y + 1)| + |x + i(y - 1)|) / 2
inline hard_work_result do_hard_work(float x, float y) {
  hard_work_result r;
  const float R = c10::metal::hypot(x, y + 1);
  const float S = c10::metal::hypot(x, y - 1);
  const float A = ::metal::max((R + S) / 2, 1.0f);
  if (A < A_CROSSOVER) {
    if (y == 1 && x < FLT_EPSILON * FLT_EPSILON / 128) {
      r.rx = precise::sqrt(x);
    } else if (x >= FLT_EPSILON * ::metal::fabs(y - 1)) {
      const float Am1 = f(x, 1 + y, R) + f(x, 1 - y, S);
      r.rx = c10::metal::log1p(Am1 + precise::sqrt(Am1 * (A + 1)));
    } else if (y < 1) {
      r.rx = x * precise::rsqrt((1 - y) * (1 + y));
    } else {
      r.rx = c10::metal::log1p((y - 1) + precise::sqrt((y - 1) * (y + 1)));
    }
  } else {
    r.rx = precise::log(A + precise::sqrt(A * A - 1));
  }
  r.new_y = y;
  if (y < FOUR_SQRT_MIN) {
    r.B_is_usable = false;
    r.sqrt_A2my2 = A * (2 / FLT_EPSILON);
    r.new_y = y * (2 / FLT_EPSILON);
    return r;
  }
  r.B = y / A;
  r.B_is_usable = r.B <= B_CROSSOVER;
  if (!r.B_is_usable) {
    if (y == 1 && x < FLT_EPSILON / 128) {
      r.sqrt_A2my2 = precise::sqrt(x) * precise::sqrt((A + y) / 2);
    } else if (x >= FLT_EPSILON * ::metal::fabs(y - 1)) {
      const float Amy = f(x, y + 1, R) + f(x, y - 1, S);
      r.sqrt_A2my2 = precise::sqrt(Amy * (A + y));
    } else if (y > 1) {
      r.sqrt_A2my2 = x * (4 / FLT_EPSILON / FLT_EPSILON) * y /
          precise::sqrt((y + 1) * (y - 1));
      r.new_y = y * (4 / FLT_EPSILON / FLT_EPSILON);
    } else {
      r.sqrt_A2my2 = precise::sqrt((1 - y) * (1 + y));
    }
  }
  return r;
}

inline float2 clog_for_large_values(float2 z) {
  const auto a = ::metal::fabs(z);
  const float re = ::metal::max(a.x, a.y) > FLT_MAX / 2
      ? precise::log(c10::metal::hypot(z.x / M_E_F, z.y / M_E_F)) + 1
      : precise::log(c10::metal::hypot(z.x, z.y));
  return float2(re, precise::atan2(z.y, z.x));
}

inline float2 casinh(float2 z) {
  const float x = z.x, y = z.y;
  const float ax = ::metal::fabs(x), ay = ::metal::fabs(y);
  if (::metal::isnan(x) || ::metal::isnan(y)) {
    if (::metal::isinf(x)) {
      return float2(x, y + y);
    }
    if (::metal::isinf(y)) {
      return float2(y, x + x);
    }
    if (y == 0) {
      return float2(x + x, y);
    }
    return float2(NAN, NAN);
  }
  if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {
    const auto w = clog_for_large_values(::metal::signbit(x) ? -z : z);
    return float2(
        ::metal::copysign(w.x + M_LN2_F, x), ::metal::copysign(w.y, y));
  }
  if (ax < SQRT_6_EPSILON / 4 && ay < SQRT_6_EPSILON / 4) {
    return z;
  }
  const auto r = do_hard_work(ax, ay);
  const float ry = r.B_is_usable ? precise::asin(r.B)
                                 : precise::atan2(r.new_y, r.sqrt_A2my2);
  return float2(::metal::copysign(r.rx, x), ::metal::copysign(ry, y));
}

inline float2 casin(float2 z) {
  const auto w = casinh(float2(z.y, z.x));
  return float2(w.y, w.x);
}

inline float2 cacos(float2 z) {
  const float x = z.x, y = z.y;
  const float ax = ::metal::fabs(x), ay = ::metal::fabs(y);
  const bool sx = ::metal::signbit(x), sy = ::metal::signbit(y);
  if (::metal::isnan(x) || ::metal::isnan(y)) {
    if (::metal::isinf(x)) {
      return float2(y + y, -INFINITY);
    }
    if (::metal::isinf(y)) {
      return float2(x + x, -y);
    }
    if (x == 0) {
      return float2(M_PI_2_F, y + y);
    }
    return float2(NAN, NAN);
  }
  if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {
    const auto w = clog_for_large_values(z);
    const float ry = w.x + M_LN2_F;
    return float2(::metal::fabs(w.y), sy ? ry : -ry);
  }
  if (x == 1 && y == 0) {
    return float2(0, -y);
  }
  if (ax < SQRT_6_EPSILON / 4 && ay < SQRT_6_EPSILON / 4) {
    return float2(M_PI_2_F - x, -y);
  }
  // Same decomposition as casinh with the roles of x and y swapped
  const auto r = do_hard_work(ay, ax);
  const float new_x = sx ? -r.new_y : r.new_y;
  const float rx = r.B_is_usable ? precise::acos(sx ? -r.B : r.B)
                                 : precise::atan2(r.sqrt_A2my2, new_x);
  return float2(rx, sy ? r.rx : -r.rx);
}

inline float2 cacosh(float2 z) {
  const auto w = cacos(z);
  if (::metal::isnan(w.x) && ::metal::isnan(w.y)) {
    return float2(w.y, w.x);
  }
  if (::metal::isnan(w.x)) {
    return float2(::metal::fabs(w.y), w.x);
  }
  if (::metal::isnan(w.y)) {
    return float2(w.y, w.y);
  }
  return float2(::metal::fabs(w.y), ::metal::copysign(w.x, z.y));
}
// x * x + y * y for x, y >= 0, avoiding underflow when y is small
inline float sum_squares(float x, float y) {
  return y < SQRT_MIN ? x * x : x * x + y * y;
}

// Re(1 / (x + iy)) = x / (x^2 + y^2), without overflow
inline float real_part_reciprocal(float x, float y) {
  const float h = c10::metal::hypot(x, y);
  return ::metal::isinf(h) ? ::metal::copysign(0.0f, x) : x / h / h;
}

inline float2 catanh(float2 z) {
  const float x = z.x, y = z.y;
  const float ax = ::metal::fabs(x), ay = ::metal::fabs(y);
  if (y == 0 && ax <= 1) {
    return float2(precise::atanh(x), y);
  }
  if (x == 0) {
    return float2(x, precise::atan(y));
  }
  if (::metal::isnan(x) || ::metal::isnan(y)) {
    if (::metal::isinf(x)) {
      return float2(::metal::copysign(0.0f, x), y + y);
    }
    if (::metal::isinf(y)) {
      return float2(::metal::copysign(0.0f, x), ::metal::copysign(M_PI_2_F, y));
    }
    return float2(NAN, NAN);
  }
  if (ax > RECIP_EPSILON || ay > RECIP_EPSILON) {
    return float2(real_part_reciprocal(x, y), ::metal::copysign(M_PI_2_F, y));
  }
  if (ax < SQRT_3_EPSILON / 2 && ay < SQRT_3_EPSILON / 2) {
    return z;
  }
  const float rx = ax == 1 && ay < FLT_EPSILON
      ? (M_LN2_F - precise::log(ay)) / 2
      : c10::metal::log1p(4 * ax / sum_squares(ax - 1, ay)) / 4;
  float ry;
  if (ax == 1) {
    ry = precise::atan2(2.0f, -ay) / 2;
  } else if (ay < FLT_EPSILON) {
    ry = precise::atan2(2 * ay, (1 - ax) * (1 + ax)) / 2;
  } else {
    ry = precise::atan2(2 * ay, (1 - ax) * (1 + ax) - ay * ay) / 2;
  }
  return float2(::metal::copysign(rx, x), ::metal::copysign(ry, y));
}

inline float2 catan(float2 z) {
  const auto w = catanh(float2(z.y, z.x));
  return float2(w.y, w.x);
}
} // namespace catrig

#define DEFINE_INVERSE_TRIG_FUNCTOR(NAME)                                      \
  struct NAME##_functor {                                                      \
    template <typename T>                                                      \
    inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(           \
        const T x) {                                                           \
      return T(precise::NAME(float(x)));                                       \
    }                                                                          \
    template <typename T>                                                      \
    inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) { \
      return precise::NAME(static_cast<float>(x));                             \
    }                                                                          \
    template <typename T>                                                      \
    inline enable_if_t<is_complex_v<T>, T> operator()(const T x) {             \
      return T(catrig::c##NAME(float2(x)));                                    \
    }                                                                          \
  }

DEFINE_INVERSE_TRIG_FUNCTOR(asin);
DEFINE_INVERSE_TRIG_FUNCTOR(acos);
DEFINE_INVERSE_TRIG_FUNCTOR(asinh);
DEFINE_INVERSE_TRIG_FUNCTOR(acosh);
DEFINE_INVERSE_TRIG_FUNCTOR(atan);
DEFINE_INVERSE_TRIG_FUNCTOR(atanh);

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
    auto magnitude = c10::metal::hypot(x.x, x.y);
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
    auto magnitude = c10::metal::hypot(x.x, x.y);
    auto real = ::precise::log10(magnitude);
    auto imag =
        (x.x == 0 && x.y == 0) ? 0 : ::precise::atan2(x.y, x.x) * M_LOG10E_F;
    return T(real, imag);
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
    return T(c10::metal::log1p(float2(x)));
  }
  inline float operator()(const bool x) {
    return x ? M_LN2_F : 0;
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
    // Base 2 complex log = ln(x+yi)/ln(2)
    auto magnitude = c10::metal::hypot(x.x, x.y);
    auto real = ::precise::log2(magnitude);
    auto imag =
        (x.x == 0 && x.y == 0) ? 0 : ::precise::atan2(x.y, x.x) * M_LOG2E_F;
    return T(real, imag);
  }
  inline float operator()(const bool x) {
    return x ? 0 : -INFINITY;
  }
};

struct lgamma_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return static_cast<T>(c10::metal::log_gamma(static_cast<float>(x)));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return c10::metal::log_gamma(static_cast<float>(x));
  }
  inline float operator()(const bool x) {
    return x ? 0 : INFINITY;
  }
};

struct digamma_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return static_cast<T>(c10::metal::digamma(static_cast<float>(x)));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return c10::metal::digamma(static_cast<float>(x));
  }
  inline float operator()(const bool x) {
    return c10::metal::digamma(static_cast<float>(x));
  }
};

struct trigamma_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(const T x) {
    return static_cast<T>(c10::metal::trigamma(static_cast<float>(x)));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(const T x) {
    return c10::metal::trigamma(static_cast<float>(x));
  }
  inline float operator()(const bool x) {
    return c10::metal::trigamma(static_cast<float>(x));
  }
};

struct polygamma_functor {
  template <typename T>
  inline enable_if_t<is_scalar_floating_point_v<T>, T> operator()(
      const T x,
      const int order) {
    return static_cast<T>(c10::metal::polygamma(order, static_cast<float>(x)));
  }
  template <typename T>
  inline enable_if_t<is_scalar_integral_v<T>, float> operator()(
      const T x,
      const int order) {
    return c10::metal::polygamma(order, static_cast<float>(x));
  }
  inline float operator()(const bool x, const int order) {
    return c10::metal::polygamma(order, static_cast<float>(x));
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
    auto coef = ::precise::pow(2, x.x);
    // y == 0: same rationale as exp_ short-circuit (avoid coef*0 = NaN).
    if (x.y == 0) {
      return T(coef, 0);
    }
    float real;
    float imag = ::precise::sincos(static_cast<float>(x.y) * M_LN2_F, real);
    using elem_t = decltype(x.x + x.x);
    return T(
        coef * static_cast<elem_t>(real), coef * static_cast<elem_t>(imag));
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
    return c10::metal::sqrt(x);
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
    // rsqrt(z) = conj(sqrt(z)) / |z|, as |sqrt(z)|^2 = |z|
    const auto s = c10::metal::sqrt(float2(x));
    const auto m = ::metal::dot(s, s);
    return T(s.x / m, -s.y / m);
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

struct frac_functor {
  template <typename T>
  inline T operator()(const T x) {
    return static_cast<T>(float(x) - trunc(float(x)));
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
DEFINE_UNARY_FLOATING_FUNCTOR(erfcx);
DEFINE_UNARY_FLOATING_FUNCTOR(erfinv);
DEFINE_UNARY_FLOATING_FUNCTOR(sinc);
DEFINE_UNARY_FLOATING_FUNCTOR(floor);
DEFINE_UNARY_FLOATING_FUNCTOR(ceil);
DEFINE_UNARY_FLOATING_FUNCTOR(trunc);

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

#define REGISTER_ROUNDING_OPS(DTYPE)      \
  REGISTER_UNARY_OP(floor, DTYPE, DTYPE); \
  REGISTER_UNARY_OP(ceil, DTYPE, DTYPE);  \
  REGISTER_UNARY_OP(trunc, DTYPE, DTYPE)

REGISTER_ROUNDING_OPS(float);
REGISTER_ROUNDING_OPS(half);
REGISTER_ROUNDING_OPS(bfloat);

REGISTER_UNARY_OP(frac, float, float);
REGISTER_UNARY_OP(frac, half, half);
REGISTER_UNARY_OP(frac, bfloat, bfloat);

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

struct logical_not_functor {
  template <typename T>
  inline bool operator()(const T x) {
    return !c10::metal::cast_to<bool>(x);
  }
};

REGISTER_UNARY_OP(logical_not, bool, bool);
REGISTER_UNARY_OP(logical_not, uchar, bool);
REGISTER_UNARY_OP(logical_not, char, bool);
REGISTER_UNARY_OP(logical_not, short, bool);
REGISTER_UNARY_OP(logical_not, int, bool);
REGISTER_UNARY_OP(logical_not, long, bool);
REGISTER_UNARY_OP(logical_not, half, bool);
REGISTER_UNARY_OP(logical_not, float, bool);
REGISTER_UNARY_OP(logical_not, bfloat, bool);
REGISTER_UNARY_OP(logical_not, float2, bool);
REGISTER_UNARY_OP(logical_not, half2, bool);

// Unsigned and bool inputs never reach the stub, see signbit_out
REGISTER_UNARY_OP(signbit, char, bool);
REGISTER_UNARY_OP(signbit, short, bool);
REGISTER_UNARY_OP(signbit, int, bool);
REGISTER_UNARY_OP(signbit, long, bool);
REGISTER_UNARY_OP(signbit, half, bool);
REGISTER_UNARY_OP(signbit, bfloat, bool);
REGISTER_UNARY_OP(signbit, float, bool);

REGISTER_UNARY_OP(abs, int, int);
REGISTER_UNARY_OP(abs, long, long);
REGISTER_UNARY_OP(abs, short, short);
REGISTER_UNARY_OP(abs, char, char);
REGISTER_UNARY_OP(abs, uchar, uchar);
REGISTER_UNARY_OP(abs, float, float);
REGISTER_UNARY_OP(abs, half, half);

#define REGISTER_SIGN_OP(DTYPE) REGISTER_UNARY_OP(sign, DTYPE, DTYPE)
REGISTER_SIGN_OP(bool);
REGISTER_SIGN_OP(uchar);
REGISTER_SIGN_OP(char);
REGISTER_SIGN_OP(short);
REGISTER_SIGN_OP(int);
REGISTER_SIGN_OP(long);
REGISTER_SIGN_OP(half);
REGISTER_SIGN_OP(bfloat);
REGISTER_SIGN_OP(float);
REGISTER_UNARY_OP(sgn, float2, float2);
REGISTER_UNARY_OP(sgn, half2, half2);

#define INSTANTIATE_UNARY_KERNELS2(DTYPE0, DTYPE1) \
  REGISTER_UNARY_OP(angle, DTYPE1, DTYPE0);        \
  REGISTER_UNARY_OP(erf, DTYPE1, DTYPE0);          \
  REGISTER_UNARY_OP(erfc, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(erfcx, DTYPE1, DTYPE0);        \
  REGISTER_UNARY_OP(erfinv, DTYPE1, DTYPE0);       \
  REGISTER_UNARY_OP(exp, DTYPE1, DTYPE0);          \
  REGISTER_UNARY_OP(expm1, DTYPE1, DTYPE0);        \
  REGISTER_UNARY_OP(sigmoid, DTYPE1, DTYPE0);      \
  REGISTER_UNARY_OP(exp2, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(log, DTYPE1, DTYPE0);          \
  REGISTER_UNARY_OP(log10, DTYPE1, DTYPE0);        \
  REGISTER_UNARY_OP(log1p, DTYPE1, DTYPE0);        \
  REGISTER_UNARY_OP(log2, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(lgamma, DTYPE1, DTYPE0);       \
  REGISTER_UNARY_OP(digamma, DTYPE1, DTYPE0);      \
  REGISTER_UNARY_OP(trigamma, DTYPE1, DTYPE0);     \
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
  REGISTER_UNARY_OP(asinh, DTYPE1, DTYPE0);        \
  REGISTER_UNARY_OP(acosh, DTYPE1, DTYPE0);        \
  REGISTER_UNARY_OP(atan, DTYPE1, DTYPE0);         \
  REGISTER_UNARY_OP(atanh, DTYPE1, DTYPE0)

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
  REGISTER_UNARY_OP(asinh, DTYPE##2, DTYPE##2);      \
  REGISTER_UNARY_OP(acosh, DTYPE##2, DTYPE##2);      \
  REGISTER_UNARY_OP(atan, DTYPE##2, DTYPE##2);       \
  REGISTER_UNARY_OP(atanh, DTYPE##2, DTYPE##2)

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

REGISTER_UNARY_ALPHA_OP(polygamma, bfloat, int, bfloat);
REGISTER_UNARY_ALPHA_OP(polygamma, half, int, half);
REGISTER_UNARY_ALPHA_OP(polygamma, float, int, float);
REGISTER_UNARY_ALPHA_OP(polygamma, bool, int, float);
REGISTER_UNARY_ALPHA_OP(polygamma, uchar, int, float);
REGISTER_UNARY_ALPHA_OP(polygamma, char, int, float);
REGISTER_UNARY_ALPHA_OP(polygamma, short, int, float);
REGISTER_UNARY_ALPHA_OP(polygamma, int, int, float);
REGISTER_UNARY_ALPHA_OP(polygamma, long, int, float);

// Replacement values are computed per-dtype on the host and ride at float
// (exact for half/bfloat extrema), mirroring the CUDA kernel's
// Scalar-to-scalar_t conversion.
inline float nan_to_num_replace(float x, NanToNumParams<float> params) {
  return ::metal::isnan(x)
      ? params.nan
      : (::metal::isinf(x) ? (x > 0.0f ? params.posinf : params.neginf) : x);
}

struct nan_to_num_functor {
  template <typename T, enable_if_t<is_scalar_floating_point_v<T>, bool> = true>
  inline T operator()(const T x, const NanToNumParams<float> params) {
    return static_cast<T>(nan_to_num_replace(float(x), params));
  }
  template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
  inline T operator()(const T x, const NanToNumParams<float> params) {
    // per-component, matching CPU/CUDA
    return T(
        nan_to_num_replace(float(x.x), params),
        nan_to_num_replace(float(x.y), params));
  }
};

typedef NanToNumParams<float> NanToNumParams_float;
REGISTER_UNARY_ALPHA_OP(nan_to_num, float, NanToNumParams_float, float);
REGISTER_UNARY_ALPHA_OP(nan_to_num, half, NanToNumParams_float, half);
REGISTER_UNARY_ALPHA_OP(nan_to_num, bfloat, NanToNumParams_float, bfloat);
REGISTER_UNARY_ALPHA_OP(nan_to_num, float2, NanToNumParams_float, float2);
REGISTER_UNARY_ALPHA_OP(nan_to_num, half2, NanToNumParams_float, half2);

// Metal's frexp() computes in float32 and Apple GPUs flush float32 subnormals
// to zero; bfloat16 shares float32's exponent range, so its subnormals are lost
// the same way. Decompose the bit pattern instead -- integer ops are not
// subject to flush-to-zero. U is T's unsigned twin.
template <typename T, typename U>
inline T frexp_bits(const T x, thread int& exponent) {
  constexpr int MANT = numeric_limits<T>::digits - 1;
  constexpr int EXP = int(sizeof(T)) * 8 - MANT - 1;
  constexpr U mant_mask = (U(1) << MANT) - 1;
  constexpr U exp_all = (U(1) << EXP) - 1;
  constexpr int bias = (1 << (EXP - 1)) - 1;

  const U bits = as_type<U>(x);
  const U sign = bits & (U(1) << (MANT + EXP));
  const U exp_field = (bits >> MANT) & exp_all;
  U frac = bits & mant_mask;

  // inf, nan and +-0 are returned unchanged with a zero exponent
  if (exp_field == exp_all || (exp_field == 0 && frac == 0)) {
    exponent = 0;
    return x;
  }

  if (exp_field == 0) {
    // Subnormal: value = frac * 2^(1 - bias - MANT). With lead the index of
    // frac's highest set bit, frac = 2^lead * 1.g, so the value is
    // 1.g * 2^(lead + 1 - bias - MANT) and the [0.5, 1) form is
    // (1.g / 2) * 2^(lead + 2 - bias - MANT). Shifting the leading one up to
    // the implicit-bit position leaves g in the significand field. frac is
    // nonzero here: the exp_field == 0 && frac == 0 case returned above, and
    // MANT + EXP is the index of U's top bit, so clz gives lead directly.
    const int lead = MANT + EXP - int(clz(frac));
    frac = (frac << (MANT - lead)) & mant_mask;
    exponent = lead + 2 - bias - MANT;
  } else {
    // Normal: value = 1.f * 2^(exp_field - bias), i.e.
    // (1.f / 2) * 2^(exp_field - bias + 1).
    exponent = int(exp_field) - bias + 1;
  }
  // An exponent field of bias-1 is 2^-1, scaling the significand into [0.5, 1).
  return as_type<T>(U(sign | (U(bias - 1) << MANT) | frac));
}

// Overload set so the kernels can stay dtype-generic, mirroring exp_ above.
inline float frexp_(const float x, thread int& e) {
  return frexp_bits<float, uint>(x, e);
}
inline half frexp_(const half x, thread int& e) {
  return frexp_bits<half, ushort>(x, e);
}
inline bfloat frexp_(const bfloat x, thread int& e) {
  return frexp_bits<bfloat, ushort>(x, e);
}

// frexp writes a mantissa and an exponent, so it cannot use the shared
// single-output unary functor path.
template <typename T>
kernel void frexp_dense(
    device T* mantissa [[buffer(0)]],
    device int* exponent [[buffer(1)]],
    device const T* input [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  int exp = 0;
  mantissa[index] = frexp_(input[index], exp);
  exponent[index] = exp;
}

// Byte strides come straight from the TensorIterator, indexed with the shared
// long-offset primitives, so nothing is downcast and arbitrarily large offsets
// are addressed the same way every other strided MPS kernel does.
template <typename T>
kernel void frexp_strided(
    device void* mantissa [[buffer(0)]],
    device void* exponent [[buffer(1)]],
    device const char* input [[buffer(2)]],
    constant long* sizes [[buffer(3)]],
    constant long* mantissa_strides [[buffer(4)]],
    constant long* exponent_strides [[buffer(5)]],
    constant long* input_strides [[buffer(6)]],
    constant uint& ndim [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  long pos[max_ndim];
  pos_from_thread_index(long(index), pos, sizes, ndim);
  int exp = 0;
  const auto m = frexp_(
      val_at_offs<T>(input, offset_from_coord(pos, input_strides, ndim)), exp);
  ref_at_offs<T>(mantissa, offset_from_coord(pos, mantissa_strides, ndim)) = m;
  ref_at_offs<int>(exponent, offset_from_coord(pos, exponent_strides, ndim)) =
      exp;
}

#define REGISTER_FREXP_OP(T)                                                \
  template [[host_name("frexp_dense_" #T)]] kernel void frexp_dense<T>(     \
      device T*, device int*, device const T*, uint);                       \
  template [[host_name("frexp_strided_" #T)]] kernel void frexp_strided<T>( \
      device void*,                                                         \
      device void*,                                                         \
      device const char*,                                                   \
      constant long*,                                                       \
      constant long*,                                                       \
      constant long*,                                                       \
      constant long*,                                                       \
      constant uint&,                                                       \
      uint)

REGISTER_FREXP_OP(float);
REGISTER_FREXP_OP(half);
REGISTER_FREXP_OP(bfloat);
