#pragma once
#include <c10/macros/Macros.h>
#include <type_traits>

namespace at {

/**
   Computes ceil(a / b)
*/
template <
    typename Res = void,
    typename T,
    typename U,
    typename = std::enable_if_t<
        std::conjunction_v<std::is_integral<T>, std::is_integral<U>>>>
C10_ALWAYS_INLINE C10_HOST_DEVICE
    std::conditional_t<std::is_same_v<Res, void>, std::common_type_t<T, U>, Res>
    ceil_div(T a, U b) {
  return (a + b - 1) / b;
}

/**
   Computes ceil(a / b) * b; i.e., rounds up `a` to the next highest
   multiple of b
*/
template <typename Res = void, typename T, typename U>
C10_ALWAYS_INLINE C10_HOST_DEVICE
    std::conditional_t<std::is_same_v<Res, void>, std::common_type_t<T, U>, Res>
    round_up(T a, U b) {
  return ceil_div(a, b) * b;
}

} // namespace at
