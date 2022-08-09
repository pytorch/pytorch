#pragma once

#include <ATen/native/special_functions/detail/spherical_legendre_y.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
spherical_legendre_y(unsigned int l, unsigned int m, T1 theta) {
  using T2 = detail::promote_t<T1>;

  return detail::spherical_legendre_y<T2>(l, m, theta);
}
} // namespace special_functions
} // namespace native
} // namespace at
