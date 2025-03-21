#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;

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

REGISTER_BINARY_INDEXING_OP(copysign, long);
REGISTER_BINARY_INDEXING_OP(copysign, int);
REGISTER_BINARY_INDEXING_OP(copysign, float);
REGISTER_BINARY_INDEXING_OP(copysign, half);
REGISTER_BINARY_INDEXING_OP(copysign, short);
REGISTER_BINARY_INDEXING_OP(copysign, uchar);
REGISTER_BINARY_INDEXING_OP(copysign, char);
REGISTER_BINARY_INDEXING_OP(copysign, bool);
REGISTER_BINARY_INDEXING_OP(fmax, float);
REGISTER_BINARY_INDEXING_OP(fmax, half);
REGISTER_BINARY_INDEXING_OP(fmin, float);
REGISTER_BINARY_INDEXING_OP(fmin, half);
REGISTER_BINARY_INDEXING_OP(nextafter, float);
REGISTER_BINARY_INDEXING_OP(nextafter, half);
REGISTER_BINARY_INDEXING_OP(zeta, float);
REGISTER_BINARY_INDEXING_OP(zeta, half);
REGISTER_BINARY_INDEXING_OP(xlog1py, float);
REGISTER_BINARY_INDEXING_OP(xlog1py, half);

#if __METAL_VERSION__ >= 310
REGISTER_BINARY_INDEXING_OP(copysign, bfloat);
REGISTER_BINARY_INDEXING_OP(fmax, bfloat);
REGISTER_BINARY_INDEXING_OP(fmin, bfloat);
REGISTER_BINARY_INDEXING_OP(nextafter, bfloat);
REGISTER_BINARY_INDEXING_OP(zeta, bfloat);
REGISTER_BINARY_INDEXING_OP(xlog1py, bfloat);
#endif

// Complex binary functions
REGISTER_BINARY_INDEXING_OP(polar, float);
REGISTER_BINARY_INDEXING_OP(polar, half);
REGISTER_BINARY_INDEXING_OP(make_complex, float);
REGISTER_BINARY_INDEXING_OP(make_complex, half);
REGISTER_BINARY_INDEXING_OP(complex_mul, float2);
REGISTER_BINARY_INDEXING_OP(complex_mul, half2);
