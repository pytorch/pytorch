#pragma once

#include <ATen/native/special_functions/detail/spherical_harmonic_y.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
std::complex<detail::promote_t<T1, T2>>
spherical_harmonic_y(unsigned int l, int m, T1 t, T2 p) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::spherical_harmonic_y<T3>(l, m, t, p);
}
} // namespace special_functions
} // namespace native
} // namespace at
