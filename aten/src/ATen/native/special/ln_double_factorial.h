#pragma once

#include <ATen/native/special/detail/ln_double_factorial.h>
#include <ATen/native/special/detail/promote_t.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
ln_double_factorial(int n) {
  using T2 = detail::promote_t<T1>;

  return detail::ln_double_factorial<T2>(n);
}
} // namespace special
} // namespace native
} // namespace at
