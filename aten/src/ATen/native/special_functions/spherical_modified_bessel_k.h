#pragma once

#include <ATen/native/special_functions/detail/spherical_modified_bessel.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
spherical_modified_bessel_k(unsigned int n, T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::spherical_modified_bessel<T2>(n, x).k;
}
} // namespace special_functions
} // namespace native
} // namespace at
