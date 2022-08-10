#pragma once

#include <ATen/native/special_functions/detail/bernoulli_polynomial_b.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr T1
bernoulli_polynomial_b(unsigned int n, T1 x) {
  return detail::bernoulli_number(n, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
