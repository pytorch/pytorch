#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/detail/tan_pi.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
tan_pi(T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::tan_pi<T2>(x);
}

template<typename T1>
inline constexpr
std::complex<T1>
tan_pi(std::complex<T1> z) {
  return detail::tan_pi(z);
}
} // namespace special_functions
} // namespace native
} // namespace at
