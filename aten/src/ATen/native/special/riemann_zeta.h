#pragma once

#include <ATen/native/special/detail/riemann_zeta.h>
#include <ATen/native/special/detail/promote_t.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
riemann_zeta(T1 s) {
  using T2 = detail::promote_t<T1>;

  return detail::riemann_zeta<T2>(s);
}
} // namespace special
} // namespace native
} // namespace at
