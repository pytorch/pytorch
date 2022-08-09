#pragma once

#include <ATen/native/special_functions/detail/laguerre_polynomial_l.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
laguerre_polynomial_l(unsigned int n, T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::laguerre_polynomial_l<T2>(n, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
