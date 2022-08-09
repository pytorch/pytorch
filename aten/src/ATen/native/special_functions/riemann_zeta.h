#pragma once

#include <ATen/native/special_functions/detail/riemann_zeta.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
riemann_zeta(T1 s) {
  using T2 = detail::promote_t<T1>;

  return detail::riemann_zeta<T2>(s);
}
} // namespace special_functions
} // namespace native
} // namespace at
