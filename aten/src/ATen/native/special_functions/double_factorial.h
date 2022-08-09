#pragma once

#include <ATen/native/special_functions/detail/double_factorial.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
double_factorial(int n) {
  using T2 = detail::promote_t<T1>;

  return detail::double_factorial<T2>(n);
}
} // namespace special_functions
} // namespace native
} // namespace at
