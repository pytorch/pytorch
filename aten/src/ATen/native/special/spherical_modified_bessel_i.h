#pragma once

#include <ATen/native/special/detail/spherical_modified_bessel.h>
#include <ATen/native/special/detail/promote_t.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
spherical_modified_bessel_i(unsigned int n, T1 z) {
  using T2 = detail::promote_t<T1>;

  return detail::spherical_modified_bessel<T2>(n, z).i;
}
} // namespace special
} // namespace native
} // namespace at
