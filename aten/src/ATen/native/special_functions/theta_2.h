#pragma once

#include "detail/elliptic_theta_2.h"
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
detail::promote_t<T1, T2>
theta_2(T1 n, T2 x) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::elliptic_theta_2<T3>(n, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
