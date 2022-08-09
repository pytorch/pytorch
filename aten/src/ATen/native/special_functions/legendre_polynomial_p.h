#pragma once

#include <ATen/native/special_functions/detail/legendre_polynomial_p.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
legendre_polynomial_p(unsigned int l, T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::legendre_polynomial_p<T2>(l, x).P_l;
}
} // namespace special_functions
} // namespace native
} // namespace at
