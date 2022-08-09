#pragma once

#include <ATen/native/special_functions/detail/carlson_elliptic_r_c.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
detail::promote_t<T1, T2>
carlson_elliptic_r_c(T1 x, T2 y) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::carlson_elliptic_r_c<T3>(x, y);
}
} // namespace special_functions
} // namespace native
} // namespace at
