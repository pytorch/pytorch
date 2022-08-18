#pragma once

#include <ATen/native/special/detail/incomplete_legendre_elliptic_integral_pi.h>
#include <ATen/native/special/detail/promote_t.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1, typename T2, typename T3>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1, T2, T3>
incomplete_legendre_elliptic_integral_pi(T1 k, T2 nu, T3 phi) {
  using T4 = detail::promote_t<T1, T2, T3>;

  return detail::incomplete_legendre_elliptic_integral_pi<T4>(k, nu, phi);
}
} // namespace special
} // namespace native
} // namespace at
