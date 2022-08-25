#pragma once

#include <ATen/native/special/detail/spherical_hankel_h_1.h>
#include <ATen/native/special/detail/promote_t.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
c10::complex<detail::promote_t<T1>>
spherical_hankel_h_1(unsigned int n, T1 z) {
  using T2 = detail::promote_t<T1>;

  return detail::spherical_hankel_h_1<T2>(n, z);
}

//template<typename T1>
//inline constexpr
//c10::complex<detail::promote_t<T1>>
//spherical_hankel_h_1(unsigned int n, c10::complex<T1> z) {
//  using T2 = detail::promote_t<T1>;
//
//  return detail::spherical_hankel_h_1<T2>(n, z);
//}
} // namespace special
} // namespace native
} // namespace at
