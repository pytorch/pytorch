#pragma once

#include <ATen/native/special_functions/detail/incomplete_elliptic_integral_pi.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2, typename T3>
inline constexpr
detail::promote_t<T1, T2, T3>
incomplete_elliptic_integral_pi(T1 k, T2 nu, T3 phi) {
  using T4 = detail::promote_t<T1, T2, T3>;

  return detail::incomplete_elliptic_integral_pi<T4>(k, nu, phi);
}
} // namespace special_functions
} // namespace native
} // namespace at
