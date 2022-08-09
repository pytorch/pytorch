#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
detail::promote_t<T1, T2>
neville_theta_n(T1 k, T2 x) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::neville_theta_n<T3>(k, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
