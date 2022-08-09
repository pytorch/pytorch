#pragma once

#include <ATen/native/special_functions/detail/spherical_bessel_j.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
spherical_bessel_j(unsigned int n, T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::spherical_bessel_j<T2>(n, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
