#pragma once

#include <ATen/native/special_functions/detail/promote.h>
#include <ATen/native/special_functions/detail/sin_pi.h>
#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
sin_pi(T1 z) {
  using T2 = detail::promote_t<T1>;

  return detail::sin_pi<T2>(z);
}

template<typename T1>
C10_HOST_DEVICE
inline constexpr
c10::complex<T1>
sin_pi(c10::complex<T1> z) {
  return detail::sin_pi(z);
}
} // namespace special
} // namespace native
} // namespace at
