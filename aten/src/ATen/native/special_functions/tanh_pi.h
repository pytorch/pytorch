#pragma once

#include <ATen/native/special_functions/detail/promote.h>
#include <ATen/native/special_functions/detail/tanh_pi.h>
#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
tanh_pi(T1 z) {
  using T2 = detail::promote_t<T1>;

  return detail::tanh_pi<T2>(z);
}

template<typename T1>
C10_HOST_DEVICE
c10::complex<T1>
tanh_pi(c10::complex<T1> z) {
  return detail::tanh_pi(z);
}
} // namespace special_functions
} // namespace native
} // namespace at
