#pragma once

#include <ATen/native/special_functions/detail/associated_laguerre_polynomial_l.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
associated_laguerre_polynomial_l(unsigned int n, unsigned int m, T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::associated_laguerre_polynomial_l<T2>(n, m, x);
}

template<typename T1, typename T2>
inline constexpr
detail::promote_t<T1, T2>
associated_laguerre_polynomial_l(unsigned int n, T1 alpha1, T2 x) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::associated_laguerre_polynomial_l<T3>(n, alpha1, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
