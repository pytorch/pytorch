#pragma once

#include <ATen/native/special/detail/promote_t.h>
#include <ATen/native/special/detail/ln_gamma.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
ln_gamma(T1 a) {
  using T2 = detail::promote_t<T1>;

  return detail::ln_gamma<T2>(a);
}

template<typename T1>
C10_HOST_DEVICE
inline constexpr
c10::complex<detail::promote_t<T1>>
ln_gamma(c10::complex<T1> a) {
  using T2 = c10::complex<detail::promote_t<T1>>;

  return detail::ln_gamma<T2>(a);
}
} // namespace special
} // namespace native
} // namespace at
