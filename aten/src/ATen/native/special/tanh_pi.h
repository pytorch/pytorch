#pragma once

#include <ATen/native/special/detail/promote_t.h>
#include <ATen/native/special/detail/tanh_pi.h>
#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

namespace at {
namespace native {
namespace special {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
tanh_pi(T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::tanh_pi<T2>(x);
}

template<typename T1>
C10_HOST_DEVICE
c10::complex<T1>
tanh_pi(c10::complex<T1> z) {
  return detail::tanh_pi(z);
}
} // namespace special
} // namespace native
} // namespace at
