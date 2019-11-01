#pragma once

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#include <cmath>
#include <type_traits>
#include <c10/util/BFloat16.h>
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

inline C10_HOST_DEVICE bool _isnan(at::BFloat16 val) {
  return at::_isnan(float(val));
}

} // namespace at
