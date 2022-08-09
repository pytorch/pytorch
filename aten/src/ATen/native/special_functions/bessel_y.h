#pragma once

#include <ATen/native/special_functions/detail/bessel_y.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
detail::promote_t<T1, T2>
bessel_y(T1 v, T2 z) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::bessel_y<T3>(z, v);
}
} // namespace special_functions
} // namespace native
} // namespace at
