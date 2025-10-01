#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;

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

struct sub_alpha_functor {
  template <typename T>
  inline T operator()(const T a, const T b, const T alpha) {
    return static_cast<T>(a - c10::metal::mul(alpha, b));
  }
};

struct lerp_alpha_functor {
  template <typename T>
  inline T operator()(const T a, const T b, const T alpha) {
    return static_cast<T>(a + c10::metal::mul(alpha, b - a));
  }
};

struct native_dropout_mask_and_scale_functor {
  template <typename TI, typename TA>
  inline TA operator()(const TI a, const TI b, const TA scale) {
    return static_cast<TA>(a) * static_cast<TA>(b) * scale;
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

struct nextafter_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(::metal::nextafter(a, b));
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
    return metal::floor(c10::metal::div(a, b));
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

#define REGISTER_INTEGER_BINARY_OP(NAME)  \
  REGISTER_BINARY_OP(NAME, long, long);   \
  REGISTER_BINARY_OP(NAME, int, int);     \
  REGISTER_BINARY_OP(NAME, short, short); \
  REGISTER_BINARY_OP(NAME, uchar, uchar); \
  REGISTER_BINARY_OP(NAME, char, char);   \
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

REGISTER_FLOAT_BINARY_OP(copysign);
REGISTER_INT2FLOAT_BINARY_OP(copysign);
REGISTER_FLOAT_BINARY_OP(fmax);
REGISTER_FLOAT_BINARY_OP(fmin);
REGISTER_FLOAT_BINARY_OP(nextafter);
REGISTER_FLOAT_BINARY_OP(zeta);
REGISTER_INT2FLOAT_BINARY_OP(zeta);
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
REGISTER_BINARY_ALPHA_OP(add_alpha, long, long, long);
REGISTER_BINARY_ALPHA_OP(add_alpha, int, int, int);
REGISTER_BINARY_ALPHA_OP(add_alpha, float, float, float);
REGISTER_BINARY_ALPHA_OP(add_alpha, half, half, half);
REGISTER_BINARY_ALPHA_OP(add_alpha, short, short, short);
REGISTER_BINARY_ALPHA_OP(add_alpha, uchar, uchar, uchar);
REGISTER_BINARY_ALPHA_OP(add_alpha, char, char, char);
REGISTER_BINARY_ALPHA_OP(add_alpha, bool, bool, bool);
REGISTER_BINARY_ALPHA_OP(sub_alpha, long, long, long);
REGISTER_BINARY_ALPHA_OP(sub_alpha, int, int, int);
REGISTER_BINARY_ALPHA_OP(sub_alpha, float, float, float);
REGISTER_BINARY_ALPHA_OP(sub_alpha, half, half, half);
REGISTER_BINARY_ALPHA_OP(sub_alpha, short, short, short);
REGISTER_BINARY_ALPHA_OP(sub_alpha, uchar, uchar, uchar);
REGISTER_BINARY_ALPHA_OP(sub_alpha, char, char, char);
REGISTER_BINARY_ALPHA_OP(sub_alpha, bool, bool, bool);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, long, long, long);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, int, int, int);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, float, float, float);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, half, half, half);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, short, short, short);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, uchar, uchar, uchar);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, char, char, char);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, bool, bool, bool);

REGISTER_BINARY_ALPHA_OP(native_dropout_mask_and_scale, float, float, float);
REGISTER_BINARY_ALPHA_OP(native_dropout_mask_and_scale, bfloat, bfloat, bfloat);
REGISTER_BINARY_ALPHA_OP(native_dropout_mask_and_scale, half, half, half);

REGISTER_BINARY_ALPHA_OP(add_alpha, bfloat, bfloat, bfloat);
REGISTER_BINARY_ALPHA_OP(sub_alpha, bfloat, bfloat, bfloat);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, bfloat, bfloat, bfloat);

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
REGISTER_BINARY_ALPHA_OP(add_alpha, float2, float2, float2);
REGISTER_BINARY_ALPHA_OP(add_alpha, half2, half2, half2);
REGISTER_BINARY_ALPHA_OP(sub_alpha, float2, float2, float2);
REGISTER_BINARY_ALPHA_OP(sub_alpha, half2, half2, half2);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, float2, float2, float2);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, half2, half2, half2);
