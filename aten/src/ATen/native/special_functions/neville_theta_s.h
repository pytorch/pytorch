#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/detail/neville_theta_s.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
detail::promote_t<T1, T2>
neville_theta_s(T1 k, T2 x) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::neville_theta_s<T3>(k, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
