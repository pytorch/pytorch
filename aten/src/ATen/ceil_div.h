#pragma once
#include <c10/macros/Macros.h>

namespace at {

/**
   Computes ceil(a / b)
*/
template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
C10_ALWAYS_INLINE C10_HOST_DEVICE T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

/**
   Computes ceil(a / b) * b; i.e., rounds up `a` to the next highest
   multiple of b
*/
template <typename T>
C10_ALWAYS_INLINE C10_HOST_DEVICE T round_up(T a, T b) {
  return ceil_div(a, b) * b;
}

} // namespace at
