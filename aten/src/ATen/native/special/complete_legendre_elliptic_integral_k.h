#pragma once

#include <ATen/native/special/detail/complete_legendre_elliptic_integral_k.h>
#include <ATen/native/special/detail/promote_t.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
complete_legendre_elliptic_integral_k(T1 k) {
  using T2 = detail::promote_t<T1>;

  return detail::complete_legendre_elliptic_integral_k<T2>(k);
}
} // namespace special
} // namespace native
} // namespace at
