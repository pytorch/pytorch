#pragma once

#include <ATen/native/special_functions/detail/spherical_hankel_h_1.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
std::complex<detail::promote_t<T1>>
spherical_hankel_h_1(unsigned int n, T1 z) {
  using T2 = detail::promote_t<T1>;

  return detail::spherical_hankel_h_1<T2>(n, z);
}

//template<typename T1>
//inline constexpr
//std::complex<detail::promote_t<T1>>
//spherical_hankel_h_1(unsigned int n, std::complex<T1> x) {
//  using T2 = detail::promote_t<T1>;
//
//  return detail::spherical_hankel_h_1<T2>(n, x);
//}
} // namespace special_functions
} // namespace native
} // namespace at
