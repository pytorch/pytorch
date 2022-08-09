#pragma once

#include <ATen/native/special_functions/detail/radial_polynomial_r.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
radial_polynomial_r(unsigned int n, unsigned int m, T1 rho) {
  using T2 = detail::promote_t<T1>;

  return detail::radial_polynomial_r<T2>(n, m, rho);
}
} // namespace special_functions
} // namespace native
} // namespace at
