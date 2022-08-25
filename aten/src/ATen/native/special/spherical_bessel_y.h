#pragma once

#include <ATen/native/special/detail/spherical_bessel_y.h>
#include <ATen/native/special/detail/promote_t.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
spherical_bessel_y(unsigned int n, T1 z) {
  using T2 = detail::promote_t<T1>;

  return detail::spherical_bessel_y<T2>(n, z);
}
} // namespace special
} // namespace native
} // namespace at
