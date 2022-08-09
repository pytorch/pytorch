#pragma once

#include <ATen/native/special_functions/detail/gegenbauer_polynomial_c.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline typename detail::promote_t<T1, T2>
gegenbauer_polynomial_c(T1 lambda, unsigned int n, T2 x) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::gegenbauer_polynomial_c<T3>(n, lambda, x).C_n;
}
} // namespace special_functions
} // namespace native
} // namespace at
