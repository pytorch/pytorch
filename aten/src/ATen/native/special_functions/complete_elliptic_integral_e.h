#pragma once

#include <ATen/native/special_functions/detail/complete_elliptic_integral_e.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
complete_elliptic_integral_e(T1 k) {
  using T2 = detail::promote_t<T1>;

  return detail::complete_elliptic_integral_e<T2>(k);
}
} // namespace special_functions
} // namespace native
} // namespace at
