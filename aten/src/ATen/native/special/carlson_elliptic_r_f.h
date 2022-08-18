#pragma once

#include <ATen/native/special/detail/carlson_elliptic_r_f.h>
#include <ATen/native/special/detail/promote_t.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1, typename T2, typename T3>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1, T2, T3>
carlson_elliptic_r_f(T1 x, T2 y, T3 z) {
  using T4 = detail::promote_t<T1, T2, T3>;

  return detail::carlson_elliptic_r_f<T4>(x, y, z);
}
} // namespace special
} // namespace native
} // namespace at
