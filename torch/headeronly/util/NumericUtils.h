#pragma once

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>
#include <torch/headeronly/util/Float8_e4m3fnuz.h>
#include <torch/headeronly/util/Float8_e5m2.h>
#include <torch/headeronly/util/Float8_e5m2fnuz.h>
#include <torch/headeronly/util/Half.h>
#include <torch/headeronly/util/complex.h>

#include <cmath>
#include <type_traits>

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)

// std::isnan isn't performant to use on integral types; it will
// (uselessly) convert to floating point and then do the test.
// This function is.

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T /*val*/) {
  return false;
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return ::isnan(val);
#else
  return std::isnan(val);
#endif
}

template <typename T, std::enable_if_t<is_complex<T>::value, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return std::isnan(val.real()) || std::isnan(val.imag());
}

template <typename T, std::enable_if_t<std::is_same_v<T, Half>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return _isnan(static_cast<float>(val));
}

template <typename T, std::enable_if_t<std::is_same_v<T, BFloat16>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(BFloat16 val) {
  return _isnan(static_cast<float>(val));
}

inline C10_HOST_DEVICE bool _isnan(BFloat16 val) {
  return _isnan(static_cast<float>(val));
}

template <typename T, std::enable_if_t<std::is_same_v<T, Float8_e5m2>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, Float8_e4m3fn>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, Float8_e5m2fnuz>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

template <
    typename T,
    std::enable_if_t<std::is_same_v<T, Float8_e4m3fnuz>, int> = 0>
inline C10_HOST_DEVICE bool _isnan(T val) {
  return val.isnan();
}

// std::isinf isn't performant to use on integral types; it will
// (uselessly) convert to floating point and then do the test.
// This function is.

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
inline C10_HOST_DEVICE bool _isinf(T /*val*/) {
  return false;
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
inline C10_HOST_DEVICE bool _isinf(T val) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return ::isinf(val);
#else
  return std::isinf(val);
#endif
}

inline C10_HOST_DEVICE bool _isinf(Half val) {
  return _isinf(static_cast<float>(val));
}

inline C10_HOST_DEVICE bool _isinf(BFloat16 val) {
  return _isinf(static_cast<float>(val));
}

inline C10_HOST_DEVICE bool _isinf(Float8_e5m2 val) {
  return val.isinf();
}

inline C10_HOST_DEVICE bool _isinf(Float8_e4m3fn val [[maybe_unused]]) {
  return false;
}

inline C10_HOST_DEVICE bool _isinf(Float8_e5m2fnuz val [[maybe_unused]]) {
  return false;
}

inline C10_HOST_DEVICE bool _isinf(Float8_e4m3fnuz val [[maybe_unused]]) {
  return false;
}

HIDDEN_NAMESPACE_END(torch, headeronly)
