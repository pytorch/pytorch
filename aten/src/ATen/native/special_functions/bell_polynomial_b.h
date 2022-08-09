#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr T2
bell_polynomial_b(unsigned int n, T2 x) {
  return detail::bell_polynomial_b(n, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
