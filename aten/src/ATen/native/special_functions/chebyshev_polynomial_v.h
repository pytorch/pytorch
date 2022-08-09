#pragma once

#include <ATen/native/special_functions/detail/chebyshev_polynomial_v.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
chebyshev_polynomial_v(unsigned int n, T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::chebyshev_polynomial_v<T2>(n, x).V_n;
}
} // namespace special_functions
} // namespace native
} // namespace at
