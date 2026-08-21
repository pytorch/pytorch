#pragma once
#include <c10/macros/Macros.h>
#include <type_traits>

namespace at {
/*
computes ceil(a / b)
*/
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
C10_ALWAYS_INLINE C10_HOST_DEVICE constexpr T ceil_div(T a, T b) {
  return a / b + static_cast<T>(a % b != 0);
}

/**
   Computes ceil(a / b) * b; i.e., rounds up `a` to the next highest
   multiple of b. Precondition: a >= 0, b > 0 (see ceil_div above).
*/
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
C10_ALWAYS_INLINE C10_HOST_DEVICE constexpr T round_up(T a, T b) {
  return ceil_div(a, b) * b;
}

} // namespace at
