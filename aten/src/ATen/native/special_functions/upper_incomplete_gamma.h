#pragma once

#include <ATen/native/special_functions/detail/upper_incomplete_gamma.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
detail::promote_t<T1, T2>
upper_incomplete_gamma(T1 a, T2 z) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::upper_incomplete_gamma<T3>(a, z);
}
} // namespace special_functions
} // namespace native
} // namespace at
