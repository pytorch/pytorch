#pragma once

#include <ATen/native/special/detail/carlson_elliptic_r_c.h>
#include <ATen/native/special/detail/promote_t.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1, typename T2>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1, T2>
carlson_elliptic_r_c(T1 x, T2 y) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::carlson_elliptic_r_c<T3>(x, y);
}
} // namespace special
} // namespace native
} // namespace at
