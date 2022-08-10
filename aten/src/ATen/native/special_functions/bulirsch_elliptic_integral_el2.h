#pragma once

#include <ATen/native/special_functions/detail/bulirsch_elliptic_integral_el2.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2, typename T3, typename T4>
inline constexpr
detail::promote_t<T1, T2, T3, T4>
bulirsch_elliptic_integral_el2(T1 x, T2 k_c, T3 a, T4 b) {
  using T5 = detail::promote_t<T1, T2, T3, T4>;

  return detail::bulirsch_elliptic_integral_el2<T5>(x, k_c, a, b);
}
} // namespace special_functions
} // namespace native
} // namespace at
