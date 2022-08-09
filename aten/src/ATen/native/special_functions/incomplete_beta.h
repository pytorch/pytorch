#pragma once

#include <ATen/native/special_functions/detail/incomplete_beta.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2, typename T3>
inline constexpr
detail::promote_t<T1, T2, T3>
incomplete_beta(T3 x, T1 a, T2 b) {
  using T4 = detail::promote_t<T1, T2, T3>;

  return detail::incomplete_beta<T4>(a, b, x);
};
} // namespace special_functions
} // namespace native
} // namespace at
