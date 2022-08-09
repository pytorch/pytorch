#pragma once

#include <ATen/native/special_functions/detail/hurwitz_zeta.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
detail::promote_t<T1, T2>
hurwitz_zeta(T1 s, T2 a) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::hurwitz_zeta<T3>(s, a);
}
} // namespace special_functions
} // namespace native
} // namespace at
