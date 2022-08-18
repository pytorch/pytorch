#pragma once

#include <ATen/native/special/detail/bulirsch_elliptic_integral_el3.h>
#include <ATen/native/special/detail/promote_t.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1, typename T2, typename T3>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1, T2, T3>
bulirsch_elliptic_integral_el3(T1 x, T2 k_c, T3 p) {
  using T4 = detail::promote_t<T1, T2, T3>;

  return detail::bulirsch_elliptic_integral_el3<T4>(x, k_c, p);
}
} // namespace special
} // namespace native
} // namespace at
