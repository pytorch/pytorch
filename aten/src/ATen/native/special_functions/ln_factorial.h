#pragma once

#include <ATen/native/special_functions/detail/ln_factorial.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
ln_factorial(unsigned int n) {
  using T2 = detail::promote_t<T1>;

  return detail::ln_factorial<T2>(n);
}
} // namespace special_functions
} // namespace native
} // namespace at
