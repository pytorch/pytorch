#pragma once

#include <ATen/NumericUtils.h>
#include <c10/util/BFloat16-math.h>

namespace at::native {

// std:: does not have clamp functors
template <typename T>
struct minimum {
  __device__ T operator()(const T& a, const T& b) const {
    if (_isnan(a)) {
      return a;
    } else if (_isnan(b)) {
      return b;
    } else if constexpr (std::is_floating_point_v<T> ||
                         c10::is_reduced_floating_point_v<T>) {
      return ::fmin(a, b);
    } else {
      return a < b ? a : b;
    }
  }
};

template <typename T>
struct maximum {
  __device__ T operator()(const T& a, const T& b) const {
    if (_isnan(a)) {
      return a;
    } else if (_isnan(b)) {
      return b;
    } else if constexpr (std::is_floating_point_v<T> ||
                         c10::is_reduced_floating_point_v<T>) {
      return ::fmax(a, b);
    } else {
      return a > b ? a : b;
    }
  }
};

} // namespace at::native
