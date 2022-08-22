#pragma once

#include <ATen/native/special/detail/promote_t.h>
#include <ATen/native/special/detail/upper_incomplete_gamma.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1, typename T2>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1, T2>
upper_incomplete_gamma(T1 a, T2 z) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::upper_incomplete_gamma<T3>(a, z);
}
} // namespace special
} // namespace native
} // namespace at
