#pragma once
#include <c10/macros/Macros.h>
#include <type_traits>

namespace at {

/**
   Computes ceil(a / b)
*/
template <
    typename N,
    typename D,
    typename = std::enable_if_t<
        std::is_integral_v<N> && std::is_integral_v<D> &&
        !std::is_same_v<N, bool> && !std::is_same_v<D, bool>>>
C10_ALWAYS_INLINE C10_HOST_DEVICE auto ceil_div(const N num, const D denom) {
  using R = decltype(num / denom);
  assert(num > 0);
  assert(denom > 0);
  const auto big_num = static_cast<R>(num);
  const auto big_denom = static_cast<R>(denom);
  return (big_num + big_denom - 1) / big_denom;
}

/**
   Computes ceil(a / b) * b; i.e., rounds up `a` to the next highest
   multiple of b
*/
template <
    typename N,
    typename D,
    typename = std::enable_if_t<
        std::is_integral_v<N> && std::is_integral_v<D> &&
        !std::is_same_v<N, bool> && !std::is_same_v<D, bool>>>
C10_ALWAYS_INLINE C10_HOST_DEVICE auto round_up(const N num, const D denom) {
  assert(num > 0);
  assert(denom > 0);
  return ceil_div(num, denom) * denom;
}

} // namespace at
