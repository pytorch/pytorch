#pragma once

#include <ATen/native/special_functions/detail/exponential_integral_e.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
exponential_integral_e(unsigned int n, T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::exponential_integral_e<T2>(n, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
