#pragma once

#include <ATen/native/special/detail/hurwitz_zeta.h>
#include <ATen/native/special/detail/promote_t.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1, typename T2>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1, T2>
hurwitz_zeta(T1 s, T2 a) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::hurwitz_zeta<T3>(s, a);
}
} // namespace special
} // namespace native
} // namespace at
