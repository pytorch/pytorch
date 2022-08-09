#pragma once

#include <ATen/native/special_functions/detail/zernike_polynomial_z.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
detail::promote_t<T1, T2>
zernike_polynomial_z(unsigned int n, int m, T1 rho, T2 phi) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::zernike_polynomial_z<T3>(n, m, rho, phi);
}
} // namespace special_functions
} // namespace native
} // namespace at
