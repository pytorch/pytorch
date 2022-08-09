#pragma once

#include <ATen/native/special_functions/detail/complete_legendre_elliptic_integral_d.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
complete_legendre_elliptic_integral_d(T1 k) {
  using T2 = detail::promote_t<T1>;

  return detail::complete_legendre_elliptic_integral_d<T2>(k);
}
} // namespace special_functions
} // namespace native
} // namespace at
