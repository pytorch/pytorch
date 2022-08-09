#pragma once

#include <ATen/native/special_functions/detail/hankel_h_1.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
std::complex<detail::promote_t<T1, T2>>
hankel_h_1(T1 v, T2 z) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::hankel_h_1<T3>(v, z);
}

//template<typename T1, typename T2>
//inline constexpr
//std::complex<detail::promote_t<T1, T2>>
//hankel_h_1(std::complex<T1> n, std::complex<T2> x) {
//  using type = detail::promote_t<T1, T2>;
//
//  return detail::hankel_h_1<type>(n, x);
//}
} // namespace special_functions
} // namespace native
} // namespace at
