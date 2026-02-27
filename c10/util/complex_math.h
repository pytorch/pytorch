#if !defined(C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H)
#error \
    "c10/util/complex_math.h is not meant to be individually included. Include c10/util/complex.h instead."
#endif

namespace c10_complex_math {

// Exponential functions

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> exp(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::exp(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::exp(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> log(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::log(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::log(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> log10(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::log10(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::log10(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> log2(const c10::complex<T>& x) {
  const c10::complex<T> log2 = c10::complex<T>(::log(2.0), 0.0);
  return c10_complex_math::log(x) / log2;
}

// Power functions
//
#if defined(_LIBCPP_VERSION) || \
    (defined(__GLIBCXX__) && !defined(_GLIBCXX11_USE_C99_COMPLEX))
namespace _detail {
C10_API c10::complex<float> sqrt(const c10::complex<float>& in);
C10_API c10::complex<double> sqrt(const c10::complex<double>& in);
C10_API c10::complex<float> acos(const c10::complex<float>& in);
C10_API c10::complex<double> acos(const c10::complex<double>& in);
} // namespace _detail
#endif

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> sqrt(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::sqrt(static_cast<thrust::complex<T>>(x)));
#elif !(                        \
    defined(_LIBCPP_VERSION) || \
    (defined(__GLIBCXX__) && !defined(_GLIBCXX11_USE_C99_COMPLEX)))
  return static_cast<c10::complex<T>>(
      std::sqrt(static_cast<std::complex<T>>(x)));
#else
  return _detail::sqrt(x);
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> pow(
    const c10::complex<T>& x,
    const c10::complex<T>& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(
      static_cast<thrust::complex<T>>(x), static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::pow(
      static_cast<std::complex<T>>(x), static_cast<std::complex<T>>(y)));
#endif
}

// Regression in ROCm 7.2. See https://github.com/ROCm/rocm-libraries/pull/3836.
// Specialized version for complex<float> on AMD GPUs to use FMA-based
// multiplication
#if defined(__HIPCC__)
namespace detail {
// FMA-aware complex multiplication for float precision on AMD GPUs.
// This prevents SLP vectorizer from breaking FMA formation, which causes
// numerical precision loss in complex arithmetic.
// The issue occurs when vectorizer packs scalar multiplies before backend
// can form FMA instructions, resulting in double rounding instead of single.
C10_HOST_DEVICE inline thrust::complex<float> complex_mul_fma(
    thrust::complex<float> a,
    thrust::complex<float> b) {
  // Complex multiplication: (a.r + a.i*i) * (b.r + b.i*i)
  // = (a.r*b.r - a.i*b.i) + (a.r*b.i + a.i*b.r)*i
  // Using __builtin_fmaf ensures FMA at source level:
  // real: a.r*b.r + (-(a.i*b.i)) = FMA(a.r, b.r, -(a.i*b.i))
  // imag: a.i*b.r + a.r*b.i = FMA(a.r, b.i, a.i*b.r)
  float real_part = __builtin_fmaf(a.real(), b.real(), -(a.imag() * b.imag()));
  float imag_part = __builtin_fmaf(a.real(), b.imag(), a.imag() * b.real());
  return thrust::complex<float>(real_part, imag_part);
}
} // namespace detail

template <>
C10_HOST_DEVICE inline c10::complex<float> pow(
    const c10::complex<float>& x,
    const c10::complex<float>& y) {
  auto log_x = thrust::log(static_cast<thrust::complex<float>>(x));
  auto y_log_x =
      detail::complex_mul_fma(static_cast<thrust::complex<float>>(y), log_x);
  return static_cast<c10::complex<float>>(thrust::exp(y_log_x));
}
#endif

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> pow(
    const c10::complex<T>& x,
    const T& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::pow(static_cast<thrust::complex<T>>(x), y));
#else
  return static_cast<c10::complex<T>>(
      std::pow(static_cast<std::complex<T>>(x), y));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> pow(
    const T& x,
    const c10::complex<T>& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::pow(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(
      std::pow(x, static_cast<std::complex<T>>(y)));
#endif
}

template <typename T, typename U>
C10_HOST_DEVICE inline c10::complex<decltype(T() * U())> pow(
    const c10::complex<T>& x,
    const c10::complex<U>& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(
      static_cast<thrust::complex<T>>(x), static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(std::pow(
      static_cast<std::complex<T>>(x), static_cast<std::complex<T>>(y)));
#endif
}

template <typename T, typename U>
C10_HOST_DEVICE inline c10::complex<decltype(T() * U())> pow(
    const c10::complex<T>& x,
    const U& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::pow(static_cast<thrust::complex<T>>(x), y));
#else
  return static_cast<c10::complex<T>>(
      std::pow(static_cast<std::complex<T>>(x), y));
#endif
}

template <typename T, typename U>
C10_HOST_DEVICE inline c10::complex<decltype(T() * U())> pow(
    const T& x,
    const c10::complex<U>& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::pow(x, static_cast<thrust::complex<T>>(y)));
#else
  return static_cast<c10::complex<T>>(
      std::pow(x, static_cast<std::complex<T>>(y)));
#endif
}

// Trigonometric functions

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> sin(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::sin(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::sin(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> cos(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::cos(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::cos(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> tan(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::tan(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::tan(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> asin(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::asin(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::asin(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> acos(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::acos(static_cast<thrust::complex<T>>(x)));
#elif !defined(_LIBCPP_VERSION)
  return static_cast<c10::complex<T>>(
      std::acos(static_cast<std::complex<T>>(x)));
#else
  return _detail::acos(x);
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> atan(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::atan(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::atan(static_cast<std::complex<T>>(x)));
#endif
}

// Hyperbolic functions

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> sinh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::sinh(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::sinh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> cosh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::cosh(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::cosh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> tanh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::tanh(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::tanh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> asinh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::asinh(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::asinh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> acosh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::acosh(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::acosh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> atanh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(
      thrust::atanh(static_cast<thrust::complex<T>>(x)));
#else
  return static_cast<c10::complex<T>>(
      std::atanh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> log1p(const c10::complex<T>& z) {
#if defined(__APPLE__) || defined(__MACOSX) || defined(__CUDACC__) || \
    defined(__HIPCC__)
  // For Mac, the new implementation yielded a high relative error. Falling back
  // to the old version for now.
  // See https://github.com/numpy/numpy/pull/22611#issuecomment-1667945354
  // For CUDA we also use this one, as thrust::log(thrust::complex) takes
  // *forever* to compile

  // log1p(z) = log(1 + z)
  // Let's define 1 + z = r * e ^ (i * a), then we have
  // log(r * e ^ (i * a)) = log(r) + i * a
  // With z = x + iy, the term r can be written as
  // r = ((1 + x) ^ 2 + y ^ 2) ^ 0.5
  //   = (1 + x ^ 2 + 2 * x + y ^ 2) ^ 0.5
  // So, log(r) is
  // log(r) = 0.5 * log(1 + x ^ 2 + 2 * x + y ^ 2)
  //        = 0.5 * log1p(x * (x + 2) + y ^ 2)
  // we need to use the expression only on certain condition to avoid overflow
  // and underflow from `(x * (x + 2) + y ^ 2)`
  T x = z.real();
  T y = z.imag();
  T zabs = std::abs(z);
  T theta = std::atan2(y, x + T(1));
  if (zabs < 0.5) {
    T r = x * (T(2) + x) + y * y;
    if (r == 0) { // handle underflow
      return {x, theta};
    }
    return {T(0.5) * std::log1p(r), theta};
  } else {
    T z0 = std::hypot(x + 1, y);
    return {std::log(z0), theta};
  }
#else
  // CPU path
  // Based on https://github.com/numpy/numpy/pull/22611#issuecomment-1667945354
  c10::complex<T> u = z + T(1);
  if (u == T(1)) {
    return z;
  } else {
    auto log_u = log(u);
    if (u - T(1) == z) {
      return log_u;
    }
    return log_u * (z / (u - T(1)));
  }
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> expm1(const c10::complex<T>& z) {
  // expm1(z) = exp(z) - 1
  // Define z = x + i * y
  // f = e ^ (x + i * y) - 1
  //   = e ^ x * e ^ (i * y) - 1
  //   = (e ^ x * cos(y) - 1) + i * (e ^ x * sin(y))
  //   = (e ^ x - 1) * cos(y) - (1 - cos(y)) + i * e ^ x * sin(y)
  //   = expm1(x) * cos(y) - 2 * sin(y / 2) ^ 2 + i * e ^ x * sin(y)
  T x = z.real();
  T y = z.imag();
  T a = std::sin(y / 2);
  T er = std::expm1(x) * std::cos(y) - T(2) * a * a;
  T ei = std::exp(x) * std::sin(y);
  return {er, ei};
}

} // namespace c10_complex_math

using c10_complex_math::acos;
using c10_complex_math::acosh;
using c10_complex_math::asin;
using c10_complex_math::asinh;
using c10_complex_math::atan;
using c10_complex_math::atanh;
using c10_complex_math::cos;
using c10_complex_math::cosh;
using c10_complex_math::exp;
using c10_complex_math::expm1;
using c10_complex_math::log;
using c10_complex_math::log10;
using c10_complex_math::log1p;
using c10_complex_math::log2;
using c10_complex_math::pow;
using c10_complex_math::sin;
using c10_complex_math::sinh;
using c10_complex_math::sqrt;
using c10_complex_math::tan;
using c10_complex_math::tanh;

namespace std {

using c10_complex_math::acos;
using c10_complex_math::acosh;
using c10_complex_math::asin;
using c10_complex_math::asinh;
using c10_complex_math::atan;
using c10_complex_math::atanh;
using c10_complex_math::cos;
using c10_complex_math::cosh;
using c10_complex_math::exp;
using c10_complex_math::expm1;
using c10_complex_math::log;
using c10_complex_math::log10;
using c10_complex_math::log1p;
using c10_complex_math::log2;
using c10_complex_math::pow;
using c10_complex_math::sin;
using c10_complex_math::sinh;
using c10_complex_math::sqrt;
using c10_complex_math::tan;
using c10_complex_math::tanh;

} // namespace std
