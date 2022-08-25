#pragma once

#include <ATen/native/special/detail/hankel_h_2.h>
#include <ATen/native/special/detail/promote_t.h>

namespace at {
namespace native {
namespace special {
template<typename T1, typename T2>
C10_HOST_DEVICE
inline constexpr
c10::complex<detail::promote_t<T1, T2>>
hankel_h_2(T1 v, T2 z) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::hankel_h_2<T3>(v, z);
}

//template<typename T1, typename T2>
//inline constexpr
//c10::complex<detail::promote_t<T1, T2>>
//hankel_h_2(c10::complex<T1> n, c10::complex<T2> x) {
//  using T3 = detail::promote_t<T1, T2>;
//
//  return detail::hankel_h_1<T3>(n, x);
//}
} // namespace special
} // namespace native
} // namespace at
