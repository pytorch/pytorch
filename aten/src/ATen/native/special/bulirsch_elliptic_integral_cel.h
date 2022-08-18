#pragma once

#include <ATen/native/special/detail/bulirsch_elliptic_integral_cel.h>
#include <ATen/native/special/detail/promote_t.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1, typename T2, typename T3, typename T4>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1, T2, T3, T4>
bulirsch_elliptic_integral_cel(T1 k_c, T2 p, T3 a, T4 b) {
  using T5 = detail::promote_t<T1, T2, T3, T4>;

  return detail::bulirsch_elliptic_integral_cel<T5>(k_c, p, a, b);
}
} // namespace special
} // namespace native
} // namespace at
