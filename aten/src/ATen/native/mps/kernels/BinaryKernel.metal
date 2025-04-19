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

struct lerp_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return static_cast<T>(b);
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

struct complex_mul_functor {
  template <typename T>
  inline T operator()(const T a, const T b) {
    return T(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
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
REGISTER_BINARY_OP(add, long, long);
REGISTER_BINARY_OP(add, int, int);
REGISTER_BINARY_OP(add, float, float);
REGISTER_BINARY_OP(add, half, half);
REGISTER_BINARY_OP(add, short, short);
REGISTER_BINARY_OP(add, uchar, uchar);
REGISTER_BINARY_OP(add, char, char);
REGISTER_BINARY_OP(add, bool, bool);
REGISTER_BINARY_OP(sub, long, long);
REGISTER_BINARY_OP(sub, int, int);
REGISTER_BINARY_OP(sub, float, float);
REGISTER_BINARY_OP(sub, half, half);
REGISTER_BINARY_OP(sub, short, short);
REGISTER_BINARY_OP(sub, uchar, uchar);
REGISTER_BINARY_OP(sub, char, char);
REGISTER_BINARY_OP(sub, bool, bool);
REGISTER_BINARY_OP(lerp, long, long);
REGISTER_BINARY_OP(lerp, int, int);
REGISTER_BINARY_OP(lerp, float, float);
REGISTER_BINARY_OP(lerp, half, half);
REGISTER_BINARY_OP(lerp, short, short);
REGISTER_BINARY_OP(lerp, uchar, uchar);
REGISTER_BINARY_OP(lerp, char, char);
REGISTER_BINARY_OP(lerp, bool, bool);

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
REGISTER_BINARY_OP(add, bfloat, bfloat);
REGISTER_BINARY_OP(sub, bfloat, bfloat);
REGISTER_BINARY_OP(lerp, bfloat, bfloat);
#endif

// Complex binary functions
REGISTER_BINARY_OP(polar, float, float2);
REGISTER_BINARY_OP(polar, half, half2);
REGISTER_BINARY_OP(make_complex, float, float2);
REGISTER_BINARY_OP(make_complex, half, half2);
REGISTER_BINARY_OP(complex_mul, float2, float2);
REGISTER_BINARY_OP(complex_mul, half2, half2);
REGISTER_BINARY_OP(add, float2, float2);
REGISTER_BINARY_OP(add, half2, half2);
REGISTER_BINARY_OP(add, long2, long2);
REGISTER_BINARY_OP(sub, float2, float2);
REGISTER_BINARY_OP(sub, half2, half2);
REGISTER_BINARY_OP(sub, long2, long2);
REGISTER_BINARY_OP(lerp, float2, float2);
REGISTER_BINARY_OP(lerp, half2, half2);
REGISTER_BINARY_OP(lerp, long2, long2);
