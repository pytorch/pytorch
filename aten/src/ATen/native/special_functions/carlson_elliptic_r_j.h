#pragma once

#include <ATen/native/special_functions/detail/carlson_elliptic_r_j.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2, typename T3, typename T4>
inline constexpr
detail::promote_t<T1, T2, T3, T4>
carlson_elliptic_r_j(T1 x, T2 y, T3 z, T4 p) {
  using T5 = detail::promote_t<T1, T2, T3, T4>;

  return detail::carlson_elliptic_r_j<T5>(x, y, z, p);
}
} // namespace special_functions
} // namespace native
} // namespace at
