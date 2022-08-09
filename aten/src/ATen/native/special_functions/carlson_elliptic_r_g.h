#pragma once

#include <ATen/native/special_functions/detail/carlson_elliptic_r_g.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2, typename T3>
inline constexpr
detail::promote_t<T1, T2, T3>
carlson_elliptic_r_g(T1 x, T2 y, T3 z) {
  using T4 = detail::promote_t<T1, T2, T3>;

  return detail::carlson_elliptic_r_g<T4>(x, y, z);
}
} // namespace special_functions
} // namespace native
} // namespace at
