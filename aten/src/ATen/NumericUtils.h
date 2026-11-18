#pragma once

#include <torch/headeronly/util/NumericUtils.h>

// complex_math.h declares ::exp/log/log1p/tan for c10::complex at global
// scope. The templates below call ::exp(x) etc., which as a qualified name is
// bound at definition, so those overloads must be visible here for
// at::exp(c10::complex<T>) to work.
#include <c10/util/complex.h>

#include <cmath>
#include <type_traits>

namespace at {

using torch::headeronly::_isinf;
using torch::headeronly::_isnan;

template <typename T>
C10_HOST_DEVICE inline T exp(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // use __expf fast approximation for peak bandwidth
  return __expf(x);
#elif defined(__SYCL_DEVICE_ONLY__)
  // use native::exp fast approximation for peak bandwidth
  return sycl::native::exp(x);
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
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // use __logf fast approximation for peak bandwidth
  return __logf(x);
#elif defined(__SYCL_DEVICE_ONLY__)
  // use native::log fast approximation for peak bandwidth
  return sycl::native::log(x);
#else
  return ::log(x);
#endif
}

template <>
C10_HOST_DEVICE inline double log<double>(double x) {
  return ::log(x);
}

template <typename T>
C10_HOST_DEVICE inline T log1p(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // use __logf fast approximation for peak bandwidth
  // NOTE: There is no __log1pf so unfortunately we lose precision.
  return __logf(1.0f + x);
#elif defined(__SYCL_DEVICE_ONLY__)
  // use native::log fast approximation for peak bandwidth
  return sycl::native::log(1.0f + x);
#else
  return ::log1p(x);
#endif
}

template <>
C10_HOST_DEVICE inline double log1p<double>(double x) {
  return ::log1p(x);
}

template <typename T>
C10_HOST_DEVICE inline T tan(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // use __tanf fast approximation for peak bandwidth
  return __tanf(x);
#elif defined(__SYCL_DEVICE_ONLY__)
  // use native::tan fast approximation for peak bandwidth
  return sycl::native::tan(x);
#else
  return ::tan(x);
#endif
}

template <>
C10_HOST_DEVICE inline double tan<double>(double x) {
  return ::tan(x);
}

} // namespace at
