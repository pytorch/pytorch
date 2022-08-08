#pragma once

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cmath>
#include <complex>
#include <type_traits>

namespace at {

// std::isnan isn't performant to use on integral types; it will
// (uselessly) convert to floating point and then do the test.
// This function is.

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline C10_HOST_DEVICE bool _isnan(T /*val*/) {
  return false;
}

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return ::isnan(val);
#else
  return std::isnan(val);
#endif
}

template <
    typename T,
    typename std::enable_if<c10::is_complex<T>::value, int>::type = 0>
inline bool _isnan(T val) {
  return std::isnan(val.real()) || std::isnan(val.imag());
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, at::Half>::value, int>::type = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return at::_isnan(static_cast<float>(val));
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, at::BFloat16>::value, int>::type =
        0>
inline C10_HOST_DEVICE bool _isnan(at::BFloat16 val) {
  return at::_isnan(static_cast<float>(val));
}

inline C10_HOST_DEVICE bool _isnan(at::BFloat16 val) {
  return at::_isnan(static_cast<float>(val));
}

// std::isinf isn't performant to use on integral types; it will
// (uselessly) convert to floating point and then do the test.
// This function is.

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline C10_HOST_DEVICE bool _isinf(T /*val*/) {
  return false;
}

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
inline C10_HOST_DEVICE bool _isinf(T val) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return ::isinf(val);
#else
  return std::isinf(val);
#endif
}

inline C10_HOST_DEVICE bool _isinf(at::Half val) {
  return at::_isinf(static_cast<float>(val));
}

inline C10_HOST_DEVICE bool _isinf(at::BFloat16 val) {
  return at::_isinf(static_cast<float>(val));
}

template <typename T>
C10_HOST_DEVICE inline T exp(T x) {
  static_assert(
      !std::is_same<T, double>::value,
      "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // use __expf fast approximation for peak bandwidth
  return __expf(x);
#else
  return ::exp(x);
#endif
}

template <>
C10_HOST_DEVICE inline double exp<double>(double x) {
  return ::exp(x);
}

template <typename T>
C10_HOST_DEVICE inline T log(T x) {
  static_assert(
      !std::is_same<T, double>::value,
      "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // use __logf fast approximation for peak bandwidth
  return __logf(x);
#else
  return ::log(x);
#endif
}

template <>
C10_HOST_DEVICE inline double log<double>(double x) {
  return ::log(x);
}

template <typename T>
C10_HOST_DEVICE inline T tan(T x) {
  static_assert(
      !std::is_same<T, double>::value,
      "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // use __tanf fast approximation for peak bandwidth
  return __tanf(x);
#else
  return ::tan(x);
#endif
}

template <>
C10_HOST_DEVICE inline double tan<double>(double x) {
  return ::tan(x);
}

} // namespace at
