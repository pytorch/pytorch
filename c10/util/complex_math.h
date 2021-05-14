#if !defined(C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H)
#error \
    "c10/util/complex_math.h is not meant to be individually included. Include c10/util/complex.h instead."
#endif

namespace c10_complex_math {

// Exponential functions

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> exp(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::exp(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x)));
#else
  return static_cast<c10::complex<T>>(
      std::exp(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> log(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::log(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x)));
#else
  return static_cast<c10::complex<T>>(
      std::log(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> log10(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::log10(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x)));
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
TORCH_API c10::complex<float> sqrt(const c10::complex<float>& in);
TORCH_API c10::complex<double> sqrt(const c10::complex<double>& in);
TORCH_API c10::complex<float> acos(const c10::complex<float>& in);
TORCH_API c10::complex<double> acos(const c10::complex<double>& in);
}; // namespace _detail
#endif

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> sqrt(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::sqrt(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x)));
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
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x),
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(y)));
#else
  return static_cast<c10::complex<T>>(std::pow(
      static_cast<std::complex<T>>(x), static_cast<std::complex<T>>(y)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> pow(
    const c10::complex<T>& x,
    const T& y) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::pow(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x), y));
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
  return static_cast<c10::complex<T>>(thrust::pow(
      x, c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(y)));
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
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x),
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(y)));
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
  return static_cast<c10::complex<T>>(thrust::pow(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x), y));
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
  return static_cast<c10::complex<T>>(thrust::pow(
      x, c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(y)));
#else
  return static_cast<c10::complex<T>>(
      std::pow(x, static_cast<std::complex<T>>(y)));
#endif
}

// Trigonometric functions

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> sin(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::sin(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x)));
#else
  return static_cast<c10::complex<T>>(
      std::sin(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> cos(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::cos(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x)));
#else
  return static_cast<c10::complex<T>>(
      std::cos(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> tan(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::tan(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x)));
#else
  return static_cast<c10::complex<T>>(
      std::tan(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> asin(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::asin(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x)));
#else
  return static_cast<c10::complex<T>>(
      std::asin(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> acos(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::acos(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x)));
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
  return static_cast<c10::complex<T>>(thrust::atan(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x)));
#else
  return static_cast<c10::complex<T>>(
      std::atan(static_cast<std::complex<T>>(x)));
#endif
}

// Hyperbolic functions

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> sinh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::sinh(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x)));
#else
  return static_cast<c10::complex<T>>(
      std::sinh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> cosh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::cosh(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x)));
#else
  return static_cast<c10::complex<T>>(
      std::cosh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> tanh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::tanh(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x)));
#else
  return static_cast<c10::complex<T>>(
      std::tanh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> asinh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::asinh(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x)));
#else
  return static_cast<c10::complex<T>>(
      std::asinh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> acosh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::acosh(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x)));
#else
  return static_cast<c10::complex<T>>(
      std::acosh(static_cast<std::complex<T>>(x)));
#endif
}

template <typename T>
C10_HOST_DEVICE inline c10::complex<T> atanh(const c10::complex<T>& x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return static_cast<c10::complex<T>>(thrust::atanh(
      c10_internal::cuda101bug_cast_c10_complex_to_thrust_complex(x)));
#else
  return static_cast<c10::complex<T>>(
      std::atanh(static_cast<std::complex<T>>(x)));
#endif
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
using c10_complex_math::log;
using c10_complex_math::log10;
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
using c10_complex_math::log;
using c10_complex_math::log10;
using c10_complex_math::log2;
using c10_complex_math::pow;
using c10_complex_math::sin;
using c10_complex_math::sinh;
using c10_complex_math::sqrt;
using c10_complex_math::tan;
using c10_complex_math::tanh;

} // namespace std
