#pragma once

#include <ATen/native/special/detail/gamma.h>
#include <ATen/native/special/detail/promote_t.h>

namespace at {
namespace native {
namespace special {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
gamma(T1 z) {
  using T2 = detail::promote_t<T1>;

  return detail::gamma<T2>(z);
}

template<typename T1>
C10_HOST_DEVICE
inline constexpr
c10::complex<detail::promote_t<T1>>
gamma(c10::complex<T1> z) {
  using T2 = c10::complex<detail::promote_t<T1>>;

  return detail::gamma<T2>(z);
}
} // namespace special
} // namespace native
} // namespace at
