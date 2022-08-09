#pragma once

#include <ATen/native/special_functions/detail/ln_rising_factorial.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
detail::promote_t<T1, T2>
ln_rising_factorial(T1 a, T2 n) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::log_rising_factorial<T3>(a, n);
}
} // namespace special_functions
} // namespace native
} // namespace at
