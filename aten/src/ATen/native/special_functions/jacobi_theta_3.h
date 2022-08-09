#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
detail::promote_t<T1, T2>
jacobi_theta_3(T1 q, T2 x) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::jacobi_theta_3<T3>(q, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
