#pragma once

#include <ATen/native/special/detail/bulirsch_elliptic_integral_el1.h>
#include <ATen/native/special/detail/promote_t.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1, typename T2>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1, T2>
bulirsch_elliptic_integral_el1(T1 x, T2 k_c) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::bulirsch_elliptic_integral_el1<T3>(x, k_c);
}
} // namespace special
} // namespace native
} // namespace at
