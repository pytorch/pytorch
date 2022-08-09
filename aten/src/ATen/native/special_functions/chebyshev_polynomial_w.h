#pragma once

#include <ATen/native/special_functions/detail/chebyshev_polynomial_w.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
chebyshev_polynomial_w(unsigned int n, T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::chebyshev_polynomial_w<T2>(n, x).W_n;
}
} // namespace special_functions
} // namespace native
} // namespace at
