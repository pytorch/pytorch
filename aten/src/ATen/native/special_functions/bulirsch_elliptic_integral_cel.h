#pragma once

#include <ATen/native/special_functions/detail/bulirsch_elliptic_integral_cel.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2, typename T3, typename T4>
inline constexpr
detail::promote_t<T1, T2, T3, T4>
bulirsch_elliptic_integral_cel(T1 k_c, T2 p, T3 a, T4 b) {
  using T5 = detail::promote_t<T1, T2, T3, T4>;

  return detail::bulirsch_elliptic_integral_cel<T5>(k_c, p, a, b);
}
} // namespace special_functions
} // namespace native
} // namespace at
