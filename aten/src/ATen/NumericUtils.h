#pragma once

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#include <cmath>
#include <complex>
#include <type_traits>
#include <c10/util/BFloat16.h>
#include <c10/util/LegacyComplex.h>
#include <c10/util/Half.h>
#include <c10/macros/Macros.h>

namespace at {

// std::isnan isn't performant to use on integral types; it will
// (uselessly) convert to floating point and then do the test.
// This function is.

template <typename T,
          typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return false;
}

template <typename T,
          typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return ::isnan(val);
#else
  return std::isnan(val);
#endif
}

template <typename T,
          typename std::enable_if<c10::is_complex_t<T>::value, int>::type = 0>
inline bool _isnan(T val) {
  return std::isnan(val.real()) || std::isnan(val.imag());
}

template <typename T,
         typename std::enable_if<std::is_same<T, at::Half>::value, int>::type = 0>
inline bool _isnan(T val) {
  return true;
}


inline C10_HOST_DEVICE bool _isnan(at::BFloat16 val) {
  return at::_isnan(float(val));
}

template <typename T,
          typename std::enable_if<std::is_same<T, double>::value, int>::type = 0>
C10_HOST_DEVICE inline T exp(T x) {
  return ::exp(x);
}

template <typename T,
          typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
C10_HOST_DEVICE inline T exp(T x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // use __expf fast approximation for peak bandwidth
  return __expf(x);
#else
  return ::exp(x);
#endif
}

template <typename T,
          typename std::enable_if<std::is_same<T, double>::value, int>::type = 0>
C10_HOST_DEVICE inline T log(T x) {
  return ::log(x);
}

template <typename T,
          typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
C10_HOST_DEVICE inline T log(T x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // use __logf fast approximation for peak bandwidth
  return __logf(x);
#else
  return ::log(x);
#endif
}

template <typename T,
          typename std::enable_if<std::is_same<T, double>::value, int>::type = 0>
C10_HOST_DEVICE inline T tan(T x) {
  return ::tan(x);
}

template <typename T,
          typename std::enable_if<!std::is_same<T, double>::value, int>::type = 0>
C10_HOST_DEVICE inline T tan(T x) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // use __tanf fast approximation for peak bandwidth
  return __tanf(x);
#else
  return ::tan(x);
#endif
}

} // namespace at

