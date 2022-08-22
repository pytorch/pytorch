#pragma once

#include <ATen/native/special/detail/promote_t.h>
#include <ATen/native/special/detail/reciprocal_gamma.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
reciprocal_gamma(T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::reciprocal_gamma<T2>(x);
}
} // namespace special
} // namespace native
} // namespace at
