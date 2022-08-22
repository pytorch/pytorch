#pragma once

#include <ATen/native/special/detail/binomial_coefficient.h>
#include <ATen/native/special/detail/promote_t.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
binomial_coefficient(unsigned int n, unsigned int k) {
  using T2 = detail::promote_t<T1>;

  return detail::binomial_coefficient<T2>(n, k);
}
} // namespace special
} // namespace native
} // namespace at
