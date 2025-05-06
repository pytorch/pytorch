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
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::zeta(a, b));
  }
};

struct xlog1py_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::xlog1py(a, b));
  }
};

struct chebyshev_polynomial_t_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::chebyshev_polynomial_t_forward(a, b));
  }
};

struct chebyshev_polynomial_u_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::chebyshev_polynomial_u_forward(a, b));
  }
};

struct chebyshev_polynomial_v_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::chebyshev_polynomial_v_forward(a, b));
  }
};

struct chebyshev_polynomial_w_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::chebyshev_polynomial_w_forward(a, b));
  }
};

struct hermite_polynomial_h_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::hermite_polynomial_h_forward(a, b));
  }
};

struct hermite_polynomial_he_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(c10::metal::hermite_polynomial_he_forward(a, b));
  }
};

struct nextafter_functor {
#if __METAL_VERSION__ < 310
  template <typename U>
  struct bit_type {};
  template <>
  struct bit_type<float> {
    using type = int;
  };
  template <>
  struct bit_type<half> {
    using type = short;
  };
#endif
  template <typename T>
  inline T operator()(const T a, const T b) {
#if __METAL_VERSION__ >= 310
    return static_cast<T>(::metal::nextafter(a, b));
#else
    using U = typename bit_type<T>::type;
    if (a == b) {
      return a;
    }
    if (::metal::isunordered(a, b)) {
      return NAN;
    }
    if (a == 0) {
      constexpr auto eps = as_type<T>(static_cast<U>(1));
      return b > 0 ? eps : -eps;
    }
    auto bits = as_type<U>(a);
    (a > 0) ^ (a > b) ? bits++ : bits--;
    return as_type<T>(bits);
#endif
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

REGISTER_BINARY_OP(copysign, long, float);
REGISTER_BINARY_OP(copysign, int, float);
REGISTER_BINARY_OP(copysign, float, float);
REGISTER_BINARY_OP(copysign, half, half);
REGISTER_BINARY_OP(copysign, short, float);
REGISTER_BINARY_OP(copysign, uchar, float);
REGISTER_BINARY_OP(copysign, char, float);
REGISTER_BINARY_OP(copysign, bool, float);
REGISTER_BINARY_OP(fmax, float, float);
REGISTER_BINARY_OP(fmax, half, half);
REGISTER_BINARY_OP(fmin, float, float);
REGISTER_BINARY_OP(fmin, half, half);
REGISTER_BINARY_OP(nextafter, float, float);
REGISTER_BINARY_OP(nextafter, half, half);
REGISTER_BINARY_OP(zeta, float, float);
REGISTER_BINARY_OP(zeta, half, half);
REGISTER_BINARY_OP(xlog1py, float, float);
REGISTER_BINARY_OP(xlog1py, half, half);
REGISTER_BINARY_OP(chebyshev_polynomial_t, float, float);
REGISTER_BINARY_OP(chebyshev_polynomial_t, half, half);
REGISTER_BINARY_OP(chebyshev_polynomial_u, float, float);
REGISTER_BINARY_OP(chebyshev_polynomial_u, half, half);
REGISTER_BINARY_OP(chebyshev_polynomial_v, float, float);
REGISTER_BINARY_OP(chebyshev_polynomial_v, half, half);
REGISTER_BINARY_OP(chebyshev_polynomial_w, float, float);
REGISTER_BINARY_OP(chebyshev_polynomial_w, half, half);
REGISTER_BINARY_OP(hermite_polynomial_h, float, float);
REGISTER_BINARY_OP(hermite_polynomial_h, half, half);
REGISTER_BINARY_OP(hermite_polynomial_he, float, float);
REGISTER_BINARY_OP(hermite_polynomial_he, half, half);
REGISTER_BINARY_OP(add, long, long);
REGISTER_BINARY_OP(add, int, int);
REGISTER_BINARY_OP(add, float, float);
REGISTER_BINARY_OP(add, half, half);
REGISTER_BINARY_OP(add, short, short);
REGISTER_BINARY_OP(add, uchar, uchar);
REGISTER_BINARY_OP(add, char, char);
REGISTER_BINARY_OP(add, bool, bool);
REGISTER_BINARY_OP(mul, long, long);
REGISTER_BINARY_OP(mul, int, int);
REGISTER_OPMATH_BINARY_OP(mul, float, float);
REGISTER_OPMATH_BINARY_OP(mul, half, half);
REGISTER_BINARY_OP(mul, short, short);
REGISTER_BINARY_OP(mul, uchar, uchar);
REGISTER_BINARY_OP(mul, char, char);
REGISTER_BINARY_OP(mul, bool, bool);
REGISTER_BINARY_OP(sub, long, long);
REGISTER_BINARY_OP(sub, int, int);
REGISTER_BINARY_OP(sub, float, float);
REGISTER_BINARY_OP(sub, half, half);
REGISTER_BINARY_OP(sub, short, short);
REGISTER_BINARY_OP(sub, uchar, uchar);
REGISTER_BINARY_OP(sub, char, char);
REGISTER_BINARY_OP(sub, bool, bool);
REGISTER_BINARY_OP(div_floor, long, long);
REGISTER_BINARY_OP(div_floor, int, int);
REGISTER_OPMATH_BINARY_OP(div_floor, float, float);
REGISTER_OPMATH_BINARY_OP(div_floor, half, half);
REGISTER_BINARY_OP(div_floor, short, short);
REGISTER_BINARY_OP(div_floor, uchar, uchar);
REGISTER_BINARY_OP(div_floor, char, char);
REGISTER_BINARY_OP(div_floor, bool, bool);
REGISTER_BINARY_OP(div_trunc, long, long);
REGISTER_BINARY_OP(div_trunc, int, int);
REGISTER_BINARY_OP(div_trunc, float, float);
REGISTER_BINARY_OP(div_trunc, half, half);
REGISTER_BINARY_OP(div_trunc, short, short);
REGISTER_BINARY_OP(div_trunc, uchar, uchar);
REGISTER_BINARY_OP(div_trunc, char, char);
REGISTER_BINARY_OP(div_trunc, bool, bool);
REGISTER_BINARY_OP(div_true, long, float);
REGISTER_BINARY_OP(div_true, int, float);
REGISTER_OPMATH_BINARY_OP(div_true, float, float);
REGISTER_OPMATH_BINARY_OP(div_true, half, half);
REGISTER_BINARY_OP(div_true, short, float);
REGISTER_BINARY_OP(div_true, uchar, float);
REGISTER_BINARY_OP(div_true, char, float);
REGISTER_BINARY_OP(div_true, bool, float);
REGISTER_BINARY_ALPHA_OP(add_alpha, long, long);
REGISTER_BINARY_ALPHA_OP(add_alpha, int, int);
REGISTER_BINARY_ALPHA_OP(add_alpha, float, float);
REGISTER_BINARY_ALPHA_OP(add_alpha, half, half);
REGISTER_BINARY_ALPHA_OP(add_alpha, short, short);
REGISTER_BINARY_ALPHA_OP(add_alpha, uchar, uchar);
REGISTER_BINARY_ALPHA_OP(add_alpha, char, char);
REGISTER_BINARY_ALPHA_OP(add_alpha, bool, bool);
REGISTER_BINARY_ALPHA_OP(sub_alpha, long, long);
REGISTER_BINARY_ALPHA_OP(sub_alpha, int, int);
REGISTER_BINARY_ALPHA_OP(sub_alpha, float, float);
REGISTER_BINARY_ALPHA_OP(sub_alpha, half, half);
REGISTER_BINARY_ALPHA_OP(sub_alpha, short, short);
REGISTER_BINARY_ALPHA_OP(sub_alpha, uchar, uchar);
REGISTER_BINARY_ALPHA_OP(sub_alpha, char, char);
REGISTER_BINARY_ALPHA_OP(sub_alpha, bool, bool);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, long, long);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, int, int);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, float, float);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, half, half);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, short, short);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, uchar, uchar);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, char, char);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, bool, bool);

#if __METAL_VERSION__ >= 310
REGISTER_BINARY_OP(copysign, bfloat, bfloat);
REGISTER_BINARY_OP(fmax, bfloat, bfloat);
REGISTER_BINARY_OP(fmin, bfloat, bfloat);
REGISTER_BINARY_OP(nextafter, bfloat, bfloat);
REGISTER_BINARY_OP(zeta, bfloat, bfloat);
REGISTER_BINARY_OP(xlog1py, bfloat, bfloat);
REGISTER_BINARY_OP(chebyshev_polynomial_t, bfloat, bfloat);
REGISTER_BINARY_OP(chebyshev_polynomial_u, bfloat, bfloat);
REGISTER_BINARY_OP(chebyshev_polynomial_v, bfloat, bfloat);
REGISTER_BINARY_OP(chebyshev_polynomial_w, bfloat, bfloat);
REGISTER_BINARY_OP(hermite_polynomial_h, bfloat, bfloat);
REGISTER_BINARY_OP(hermite_polynomial_he, bfloat, bfloat);
REGISTER_BINARY_OP(add, bfloat, bfloat);
REGISTER_OPMATH_BINARY_OP(mul, bfloat, bfloat);
REGISTER_BINARY_OP(sub, bfloat, bfloat);
REGISTER_OPMATH_BINARY_OP(div_floor, bfloat, bfloat);
REGISTER_BINARY_OP(div_trunc, bfloat, bfloat);
REGISTER_OPMATH_BINARY_OP(div_true, bfloat, bfloat);
REGISTER_BINARY_ALPHA_OP(add_alpha, bfloat, bfloat);
REGISTER_BINARY_ALPHA_OP(sub_alpha, bfloat, bfloat);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, bfloat, bfloat);
#endif

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
REGISTER_BINARY_ALPHA_OP(add_alpha, float2, float2);
REGISTER_BINARY_ALPHA_OP(add_alpha, half2, half2);
REGISTER_BINARY_ALPHA_OP(sub_alpha, float2, float2);
REGISTER_BINARY_ALPHA_OP(sub_alpha, half2, half2);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, float2, float2);
REGISTER_BINARY_ALPHA_OP(lerp_alpha, half2, half2);
