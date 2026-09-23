#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;
using c10::metal::float8_e4m3fn;

struct add_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(a + b);
  }
};

struct sub_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(a - b);
  }
};

struct add_alpha_functor {
  template <typename T>
  inline T operator()(const T a, const T b, const T alpha) {
    return static_cast<T>(a + c10::metal::mul(alpha, b));
  }
};

struct lerp_alpha_functor {
  // Computed at opmath precision, matching CPU/CUDA: a low-precision `alpha`
  // both loses accuracy here and cannot represent the weights `lerp.Scalar`
  // accepts (e.g. 70000).
  template <typename T, typename A>
  inline T operator()(const T a, const T b, const A alpha) {
    using op_t = c10::metal::opmath_t<T>;
    return static_cast<T>(
        op_t(a) + c10::metal::mul(static_cast<op_t>(alpha), op_t(b) - op_t(a)));
  }
};

struct fmax_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(::metal::fmax(a, b));
  }
};

struct fmin_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(::metal::fmin(a, b));
  }
};

struct maximum_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return c10::metal::max(a, b);
  }
};

struct minimum_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return c10::metal::min(a, b);
  }
};

struct copysign_functor {
  template <typename T>
  inline enable_if_t<is_floating_point_v<T>, T> operator()(
      const T a,
      const T b) {
    return static_cast<T>(::metal::copysign(a, b));
  }
  template <typename T>
  inline enable_if_t<!is_floating_point_v<T>, float> operator()(
      const T a,
      const T b) {
    return ::metal::copysign(static_cast<float>(a), static_cast<float>(b));
  }
};

struct zeta_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::zeta(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::zeta(float(a), float(b));
  }
};

struct logaddexp_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return c10::metal::logaddexp(a, b);
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::logaddexp(float(a), float(b));
  }
};

struct logaddexp2_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return c10::metal::logaddexp2(a, b);
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::logaddexp2(float(a), float(b));
  }
};

struct xlogy_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::xlogy(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::xlogy(float(a), float(b));
  }
  inline float operator()(const bool a, const bool b) {
    return (a && !b) ? -INFINITY : 0;
  }
};

struct xlog1py_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::xlog1py(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::xlog1py(float(a), float(b));
  }
};

struct chebyshev_polynomial_t_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::chebyshev_polynomial_t_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::chebyshev_polynomial_t_forward(float(a), float(b));
  }
};

struct chebyshev_polynomial_u_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::chebyshev_polynomial_u_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::chebyshev_polynomial_u_forward(float(a), float(b));
  }
};

struct chebyshev_polynomial_v_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::chebyshev_polynomial_v_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::chebyshev_polynomial_v_forward(float(a), float(b));
  }
};

struct chebyshev_polynomial_w_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::chebyshev_polynomial_w_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::chebyshev_polynomial_w_forward(float(a), float(b));
  }
};

struct shifted_chebyshev_polynomial_t_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(
        c10::metal::shifted_chebyshev_polynomial_t_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::shifted_chebyshev_polynomial_t_forward(
        float(a), float(b));
  }
};

struct shifted_chebyshev_polynomial_u_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(
        c10::metal::shifted_chebyshev_polynomial_u_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::shifted_chebyshev_polynomial_u_forward(
        float(a), float(b));
  }
};

struct shifted_chebyshev_polynomial_v_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(
        c10::metal::shifted_chebyshev_polynomial_v_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::shifted_chebyshev_polynomial_v_forward(
        float(a), float(b));
  }
};

struct shifted_chebyshev_polynomial_w_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(
        c10::metal::shifted_chebyshev_polynomial_w_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::shifted_chebyshev_polynomial_w_forward(
        float(a), float(b));
  }
};

struct hermite_polynomial_h_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::hermite_polynomial_h_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::hermite_polynomial_h_forward(float(a), float(b));
  }
};

struct hermite_polynomial_he_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::hermite_polynomial_he_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::hermite_polynomial_he_forward(float(a), float(b));
  }
};

struct laguerre_polynomial_l_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::laguerre_polynomial_l_forward(a, b));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::laguerre_polynomial_l_forward(float(a), float(b));
  }
};

struct nextafter_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(::metal::nextafter(a, b));
  }

  static inline ushort nextafter_bfloat_bits(
      const bfloat from,
      const bfloat to) {
    ushort uf = as_type<ushort>(from);
    const ushort ut = as_type<ushort>(to);
    const ushort af = uf & 0x7fff;
    const ushort at = ut & 0x7fff;

    if (uf == ut || (af == 0 && at == 0)) {
      return ut;
    }
    if (af == 0) {
      return ushort((ut & 0x8000) | 1);
    }

    const bool neg = (uf & 0x8000) != 0;
    const int from_value = neg ? -int(af) : int(af);
    const int to_value = (ut & 0x8000) ? -int(at) : int(at);
    uf += ((from_value < to_value) != neg) ? 1 : -1;
    return uf;
  }

  inline bfloat operator()(const bfloat from, const bfloat to) {
    ushort result;
    if (from != from || to != to) {
      result = as_type<ushort>(bfloat(from + to));
    } else {
      result = nextafter_bfloat_bits(from, to);
    }
    return as_type<bfloat>(result);
  }
};

struct hypot_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(
        c10::metal::hypot(::metal::fabs(a), ::metal::fabs(b)));
  }
};

struct atan2_functor {
  template <typename T, enable_if_t<is_floating_point_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(precise::atan2(float(a), float(b)));
  }
  template <typename T, enable_if_t<is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return precise::atan2(float(a), float(b));
  }
};

struct pow_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::pow(a, b));
  }
};

// Complex binary functors
struct polar_functor {
  template <typename U>
  using ret_type = c10::metal::vec2type_t<U>;
  template <typename T>
  inline ret_type<T> operator()(const T a, const T b) {
    return ret_type<T>(a * cos(b), a * sin(b));
  }
};

// Constructs complex tensor from real and imaginary planes
struct make_complex_functor {
  template <typename U>
  using ret_type = c10::metal::vec2type_t<U>;
  template <typename T>
  inline ret_type<T> operator()(const T a, const T b) {
    return ret_type<T>(a, b);
  }
};

struct mul_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return c10::metal::mul(a, b);
  }
};

struct bitwise_and_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return a & b;
  }
};

struct bitwise_or_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return a | b;
  }
};

struct bitwise_xor_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return a ^ b;
  }
};

struct bitwise_left_shift_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return a << b;
  }
};

struct bitwise_right_shift_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return a >> b;
  }
};

struct div_true_functor {
  template <
      typename T,
      ::metal::enable_if_t<!::metal::is_integral_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return c10::metal::div(a, b);
  }
  template <
      typename T,
      ::metal::enable_if_t<::metal::is_integral_v<T>, bool> = true>
  inline float operator()(const T a, const T b) {
    return c10::metal::div(float(a), float(b));
  }
};

struct div_floor_functor {
  template <
      typename T,
      ::metal::enable_if_t<!::metal::is_integral_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return c10::metal::div_floor(a, b);
  }
  template <
      typename T,
      ::metal::enable_if_t<
          ::metal::is_integral_v<T>&& ::metal::is_signed_v<T>,
          bool> = true>
  inline T operator()(const T a, const T b) {
    const auto quot = a / b;
    if ((a < 0) == (b < 0)) {
      return quot;
    }
    return a % b != 0 ? quot - 1 : quot;
  }
  template <
      typename T,
      ::metal::enable_if_t<
          ::metal::is_integral_v<T> && !::metal::is_signed_v<T>,
          bool> = true>
  inline T operator()(const T a, const T b) {
    return a / b;
  }
};

struct div_trunc_functor {
  template <
      typename T,
      ::metal::enable_if_t<!::metal::is_integral_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return T(metal::trunc(c10::metal::div(a, b)));
  }
  template <
      typename T,
      ::metal::enable_if_t<::metal::is_integral_v<T>, bool> = true>
  inline T operator()(const T a, const T b) {
    return a / b;
  }
};

struct remainder_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return T(c10::metal::remainder(a, b));
  }
};

struct fmod_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return c10::metal::fmod(a, b);
  }
};

struct igamma_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return c10::metal::igamma(a, b);
  }
};

struct igammac_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return c10::metal::igammac(a, b);
  }
};

struct gcd_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    // Euclidean algorithm for GCD
    T x = a < 0 ? -a : a;
    T y = b < 0 ? -b : b;
    while (x != 0) {
      T c = x;
      x = y % x;
      y = c;
    }
    return y;
  }
};

struct lcm_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    T g = gcd_functor{}(a, b);
    if (g == 0) {
      return 0;
    }
    // `auto` keeps the C++ integer-promoted type (sub-int types widen to int),
    // so the abs matches the CPU/CUDA kernels: abs is taken before narrowing
    // back to T on return. Divide before multiplying to limit overflow.
    auto r = a / g * b;
    return ::metal::abs(r);
  }
};

// eq/ne are defined manually (rather than via DEFINE_BINARY_COMPARISON_FUNCTOR)
// so they can carry complex overloads: `float2 == float2` returns `bool2` in
// Metal, which doesn't implicitly convert to bool. The reduction `all(...)` /
// `any(...)` is the equivalent of componentwise (real && imag) compare.
struct eq_functor {
  template <typename T>
  inline bool operator()(const T a, const T b) {
    return a == b;
  }
  inline bool operator()(const float2 a, const float2 b) {
    return all(a == b);
  }
  inline bool operator()(const half2 a, const half2 b) {
    return all(a == b);
  }
};
struct ne_functor {
  template <typename T>
  inline bool operator()(const T a, const T b) {
    return a != b;
  }
  inline bool operator()(const float2 a, const float2 b) {
    return any(a != b);
  }
  inline bool operator()(const half2 a, const half2 b) {
    return any(a != b);
  }
};
DEFINE_BINARY_COMPARISON_FUNCTOR(lt, <);
DEFINE_BINARY_COMPARISON_FUNCTOR(le, <=);
DEFINE_BINARY_COMPARISON_FUNCTOR(gt, >);
DEFINE_BINARY_COMPARISON_FUNCTOR(ge, >=);

// Logical ops test truthiness of each operand then combine. cast_to<bool>
// handles every dtype: scalars as x != 0, complex as the per-component nonzero
// test.
struct logical_and_functor {
  template <typename T>
  inline bool operator()(const T a, const T b) {
    return c10::metal::cast_to<bool>(a) && c10::metal::cast_to<bool>(b);
  }
};
struct logical_or_functor {
  template <typename T>
  inline bool operator()(const T a, const T b) {
    return c10::metal::cast_to<bool>(a) || c10::metal::cast_to<bool>(b);
  }
};
struct logical_xor_functor {
  template <typename T>
  inline bool operator()(const T a, const T b) {
    return c10::metal::cast_to<bool>(a) != c10::metal::cast_to<bool>(b);
  }
};

#define REGISTER_INTEGER_BINARY_OP_NO_BOOL(NAME) \
  REGISTER_BINARY_OP(NAME, long, long);          \
  REGISTER_BINARY_OP(NAME, int, int);            \
  REGISTER_BINARY_OP(NAME, short, short);        \
  REGISTER_BINARY_OP(NAME, uchar, uchar);        \
  REGISTER_BINARY_OP(NAME, char, char)

#define REGISTER_INTEGER_BINARY_OP(NAME)    \
  REGISTER_INTEGER_BINARY_OP_NO_BOOL(NAME); \
  REGISTER_BINARY_OP(NAME, bool, bool)

#define REGISTER_INT2FLOAT_BINARY_OP(NAME) \
  REGISTER_BINARY_OP(NAME, long, float);   \
  REGISTER_BINARY_OP(NAME, int, float);    \
  REGISTER_BINARY_OP(NAME, short, float);  \
  REGISTER_BINARY_OP(NAME, uchar, float);  \
  REGISTER_BINARY_OP(NAME, char, float);   \
  REGISTER_BINARY_OP(NAME, bool, float)

#define REGISTER_FLOAT_BINARY_OP(NAME)    \
  REGISTER_BINARY_OP(NAME, float, float); \
  REGISTER_BINARY_OP(NAME, half, half);   \
  REGISTER_BINARY_OP(NAME, bfloat, bfloat)

#define REGISTER_OPMATH_FLOAT_BINARY_OP(NAME)    \
  REGISTER_OPMATH_BINARY_OP(NAME, float, float); \
  REGISTER_OPMATH_BINARY_OP(NAME, half, half);   \
  REGISTER_OPMATH_BINARY_OP(NAME, bfloat, bfloat)

// Comparison ops produce bool but may be invoked with a non-bool `out=`
// (e.g. `linalg_vector_norm(p=0)` -> `ne_outf(float, 0, float_out)`); each
// DTYPEI gets the castout variant so the dispatcher can route the non-bool
// case through `_strided_castout_<bool>_<DTYPEI>`.
#define REGISTER_COMPARISON_OP(NAME)              \
  REGISTER_BINARY_OP(NAME, float, bool);          \
  REGISTER_BINARY_CASTOUT_OP(NAME, float, bool);  \
  REGISTER_BINARY_OP(NAME, half, bool);           \
  REGISTER_BINARY_CASTOUT_OP(NAME, half, bool);   \
  REGISTER_BINARY_OP(NAME, bfloat, bool);         \
  REGISTER_BINARY_CASTOUT_OP(NAME, bfloat, bool); \
  REGISTER_BINARY_OP(NAME, long, bool);           \
  REGISTER_BINARY_CASTOUT_OP(NAME, long, bool);   \
  REGISTER_BINARY_OP(NAME, int, bool);            \
  REGISTER_BINARY_CASTOUT_OP(NAME, int, bool);    \
  REGISTER_BINARY_OP(NAME, short, bool);          \
  REGISTER_BINARY_CASTOUT_OP(NAME, short, bool);  \
  REGISTER_BINARY_OP(NAME, uchar, bool);          \
  REGISTER_BINARY_CASTOUT_OP(NAME, uchar, bool);  \
  REGISTER_BINARY_OP(NAME, char, bool);           \
  REGISTER_BINARY_CASTOUT_OP(NAME, char, bool);   \
  REGISTER_BINARY_OP(NAME, bool, bool);           \
  REGISTER_BINARY_CASTOUT_OP(NAME, bool, bool)

// Complex variants for eq/ne and the logical ops (complex->bool). lt/le/gt/ge
// are not well-defined on complex numbers, so they don't use this.
#define REGISTER_COMPLEX_EQ_OP(NAME)              \
  REGISTER_BINARY_OP(NAME, float2, bool);         \
  REGISTER_BINARY_CASTOUT_OP(NAME, float2, bool); \
  REGISTER_BINARY_OP(NAME, half2, bool);          \
  REGISTER_BINARY_CASTOUT_OP(NAME, half2, bool)

#define REGISTER_FP8_EQ_OP(NAME)                 \
  REGISTER_BINARY_OP(NAME, float8_e4m3fn, bool); \
  REGISTER_BINARY_CASTOUT_OP(NAME, float8_e4m3fn, bool)

REGISTER_FLOAT_BINARY_OP(hypot);
REGISTER_FLOAT_BINARY_OP(atan2);
REGISTER_INT2FLOAT_BINARY_OP(atan2);
REGISTER_FLOAT_BINARY_OP(copysign);
REGISTER_INT2FLOAT_BINARY_OP(copysign);
REGISTER_FLOAT_BINARY_OP(fmax);
REGISTER_FLOAT_BINARY_OP(fmin);
REGISTER_FLOAT_BINARY_OP(maximum);
REGISTER_INTEGER_BINARY_OP(maximum);
REGISTER_FLOAT_BINARY_OP(minimum);
REGISTER_INTEGER_BINARY_OP(minimum);
REGISTER_FLOAT_BINARY_OP(nextafter);
REGISTER_FLOAT_BINARY_OP(zeta);
REGISTER_INT2FLOAT_BINARY_OP(zeta);
REGISTER_FLOAT_BINARY_OP(logaddexp);
REGISTER_INT2FLOAT_BINARY_OP(logaddexp);
REGISTER_FLOAT_BINARY_OP(logaddexp2);
REGISTER_INT2FLOAT_BINARY_OP(logaddexp2);
REGISTER_FLOAT_BINARY_OP(xlogy);
REGISTER_INT2FLOAT_BINARY_OP(xlogy);
REGISTER_FLOAT_BINARY_OP(xlog1py);
REGISTER_INT2FLOAT_BINARY_OP(xlog1py);
REGISTER_FLOAT_BINARY_OP(chebyshev_polynomial_t);
REGISTER_INT2FLOAT_BINARY_OP(chebyshev_polynomial_t);
REGISTER_FLOAT_BINARY_OP(chebyshev_polynomial_u);
REGISTER_INT2FLOAT_BINARY_OP(chebyshev_polynomial_u);
REGISTER_FLOAT_BINARY_OP(chebyshev_polynomial_v);
REGISTER_INT2FLOAT_BINARY_OP(chebyshev_polynomial_w);
REGISTER_FLOAT_BINARY_OP(chebyshev_polynomial_w);
REGISTER_INT2FLOAT_BINARY_OP(chebyshev_polynomial_v);
REGISTER_FLOAT_BINARY_OP(shifted_chebyshev_polynomial_t);
REGISTER_INT2FLOAT_BINARY_OP(shifted_chebyshev_polynomial_t);
REGISTER_FLOAT_BINARY_OP(shifted_chebyshev_polynomial_u);
REGISTER_INT2FLOAT_BINARY_OP(shifted_chebyshev_polynomial_u);
REGISTER_FLOAT_BINARY_OP(shifted_chebyshev_polynomial_v);
REGISTER_INT2FLOAT_BINARY_OP(shifted_chebyshev_polynomial_v);
REGISTER_FLOAT_BINARY_OP(shifted_chebyshev_polynomial_w);
REGISTER_INT2FLOAT_BINARY_OP(shifted_chebyshev_polynomial_w);
REGISTER_FLOAT_BINARY_OP(hermite_polynomial_h);
REGISTER_INT2FLOAT_BINARY_OP(hermite_polynomial_h);
REGISTER_FLOAT_BINARY_OP(hermite_polynomial_he);
REGISTER_INT2FLOAT_BINARY_OP(hermite_polynomial_he);
REGISTER_FLOAT_BINARY_OP(laguerre_polynomial_l);
REGISTER_INT2FLOAT_BINARY_OP(laguerre_polynomial_l);
REGISTER_FLOAT_BINARY_OP(pow);
REGISTER_INTEGER_BINARY_OP(pow);
REGISTER_BINARY_OP(pow, float2, float2);
// chalf pow must accumulate in float2: the polar form exp(y*log(x)) overflows
// half for moderately large magnitudes (e.g. (300+0j)**1 -> inf), see #195585
REGISTER_OPMATH_BINARY_OP(pow, half2, half2);
REGISTER_FLOAT_BINARY_OP(add);
REGISTER_INTEGER_BINARY_OP(add);
REGISTER_OPMATH_FLOAT_BINARY_OP(mul);
REGISTER_INTEGER_BINARY_OP(mul);
REGISTER_FLOAT_BINARY_OP(sub);
REGISTER_INTEGER_BINARY_OP(sub);
REGISTER_OPMATH_FLOAT_BINARY_OP(div_floor);
REGISTER_INTEGER_BINARY_OP(div_floor);
REGISTER_FLOAT_BINARY_OP(div_trunc);
REGISTER_INTEGER_BINARY_OP(div_trunc);
REGISTER_OPMATH_FLOAT_BINARY_OP(div_true);
REGISTER_INT2FLOAT_BINARY_OP(div_true);
REGISTER_OPMATH_FLOAT_BINARY_OP(remainder);
REGISTER_INTEGER_BINARY_OP(remainder);
REGISTER_OPMATH_FLOAT_BINARY_OP(fmod);
REGISTER_INTEGER_BINARY_OP(fmod);
REGISTER_OPMATH_FLOAT_BINARY_OP(igamma);
REGISTER_OPMATH_FLOAT_BINARY_OP(igammac);
REGISTER_INTEGER_BINARY_OP(gcd);
REGISTER_INTEGER_BINARY_OP(lcm);
REGISTER_INTEGER_BINARY_OP(bitwise_and);
REGISTER_INTEGER_BINARY_OP(bitwise_or);
REGISTER_INTEGER_BINARY_OP(bitwise_xor);
REGISTER_INTEGER_BINARY_OP_NO_BOOL(bitwise_left_shift);
REGISTER_INTEGER_BINARY_OP_NO_BOOL(bitwise_right_shift);
REGISTER_COMPARISON_OP(eq);
REGISTER_COMPLEX_EQ_OP(eq);
REGISTER_FP8_EQ_OP(eq);
REGISTER_COMPARISON_OP(ne);
REGISTER_COMPLEX_EQ_OP(ne);
REGISTER_FP8_EQ_OP(ne);
REGISTER_COMPARISON_OP(lt);
REGISTER_COMPARISON_OP(le);
REGISTER_COMPARISON_OP(gt);
REGISTER_COMPARISON_OP(ge);
REGISTER_COMPARISON_OP(logical_and);
REGISTER_COMPLEX_EQ_OP(logical_and);
REGISTER_COMPARISON_OP(logical_or);
REGISTER_COMPLEX_EQ_OP(logical_or);
REGISTER_COMPARISON_OP(logical_xor);
REGISTER_COMPLEX_EQ_OP(logical_xor);
REGISTER_BINARY_ALPHA_OP(add_alpha, long, long, long);
REGISTER_BINARY_ALPHA_OP(add_alpha, int, int, int);
REGISTER_BINARY_ALPHA_OP(add_alpha, float, float, float);
REGISTER_BINARY_ALPHA_OP(add_alpha, half, half, half);
REGISTER_BINARY_ALPHA_OP(add_alpha, short, short, short);
REGISTER_BINARY_ALPHA_OP(add_alpha, uchar, uchar, uchar);
REGISTER_BINARY_ALPHA_OP(add_alpha, char, char, char);
REGISTER_BINARY_ALPHA_OP(add_alpha, bool, bool, bool);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, long, long, long);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, int, int, int);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, float, float, float);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, half, float, half);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, short, short, short);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, uchar, uchar, uchar);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, char, char, char);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, bool, bool, bool);

REGISTER_BINARY_ALPHA_OP(add_alpha, bfloat, bfloat, bfloat);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, bfloat, float, bfloat);

// Complex binary functions
REGISTER_BINARY_OP(polar, float, float2);
REGISTER_BINARY_OP(polar, half, half2);
REGISTER_BINARY_OP(make_complex, float, float2);
REGISTER_BINARY_OP(make_complex, half, half2);
REGISTER_OPMATH_BINARY_OP(mul, float2, float2);
REGISTER_OPMATH_BINARY_OP(mul, half2, half2);
REGISTER_OPMATH_BINARY_OP(div_true, float2, float2);
REGISTER_OPMATH_BINARY_OP(div_true, half2, half2);
REGISTER_BINARY_OP(add, float2, float2);
REGISTER_BINARY_OP(add, half2, half2);
REGISTER_BINARY_OP(sub, float2, float2);
REGISTER_BINARY_OP(sub, half2, half2);
REGISTER_BINARY_OP(logaddexp, float2, float2);
REGISTER_BINARY_OP(logaddexp, half2, half2);
REGISTER_BINARY_ALPHA_OP(add_alpha, float2, float2, float2);
REGISTER_BINARY_ALPHA_OP(add_alpha, half2, half2, half2);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, float2, float2, float2);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, half2, half2, half2);

// lerp with tensor weight: lerp(s, e, w) = s + w * (e - s)
// Computed at opmath precision, matching the CPU/CUDA reference, which casts
// all three operands to `opmath_type` before interpolating.
struct lerp_functor {
  template <typename T>
  inline T operator()(const T s, const T e, const T w) {
    return static_cast<T>(s + c10::metal::mul(w, e - s));
  }
};

REGISTER_OPMATH_TERNARY_OP(lerp, float, float);
REGISTER_OPMATH_TERNARY_OP(lerp, half, half);
REGISTER_OPMATH_TERNARY_OP(lerp, bfloat, bfloat);
REGISTER_OPMATH_TERNARY_OP(lerp, float2, float2);
REGISTER_OPMATH_TERNARY_OP(lerp, long, long);
