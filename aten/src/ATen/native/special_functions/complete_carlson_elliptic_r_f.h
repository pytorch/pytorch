#pragma once

#include <ATen/native/special_functions/detail/complete_carlson_elliptic_r_f.h>
#include <ATen/native/special_functions/detail/promote_t.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1, T2>
complete_carlson_elliptic_r_f(T1 x, T2 y) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::complete_carlson_elliptic_r_f<T3>(x, y);
}
} // namespace special_functions
} // namespace native
} // namespace at
