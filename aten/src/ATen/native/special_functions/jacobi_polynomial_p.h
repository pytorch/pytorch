#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/detail/jacobi_polynomial_p.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2, typename T3>
inline constexpr
detail::promote_t<T1, T2, T3>
jacobi_polynomial_p(T1 alpha, T2 beta, unsigned n, T3 x) {
  using T4 = detail::promote_t<T1, T2, T3>;

  return detail::jacobi_polynomial_p<T4>(n, alpha, beta, x).P_n;
}
} // namespace special_functions
} // namespace native
} // namespace at
