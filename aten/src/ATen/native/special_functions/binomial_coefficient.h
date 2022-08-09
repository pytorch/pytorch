#pragma once

#include <ATen/native/special_functions/detail/binomial_coefficient.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
binomial_coefficient(unsigned int n, unsigned int k) {
  using T2 = detail::promote_t<T1>;

  return detail::binomial_coefficient<T2>(n, k);
}
} // namespace special_functions
} // namespace native
} // namespace at
