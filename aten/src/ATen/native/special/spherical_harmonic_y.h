#pragma once

#include <ATen/native/special/detail/spherical_harmonic_y.h>
#include <ATen/native/special/detail/promote_t.h>

namespace at {
namespace native {
namespace special {
template<typename T1, typename T2>
C10_HOST_DEVICE
inline constexpr
std::complex<detail::promote_t<T1, T2>>
spherical_harmonic_y(unsigned int l, int m, T1 theta, T2 phi) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::spherical_harmonic_y<T3>(l, m, theta, phi);
}
} // namespace special
} // namespace native
} // namespace at
