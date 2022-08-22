#pragma once

#include <ATen/native/special/detail/falling_factorial.h>
#include <ATen/native/special/detail/promote_t.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1, typename T2>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1, T2>
falling_factorial(T1 a, T2 n) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::falling_factorial<T3>(a, n);
}
} // namespace special
} // namespace native
} // namespace at
