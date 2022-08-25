#pragma once

#include <ATen/native/special/detail/exp_modified_bessel_k.h>
#include <ATen/native/special/detail/promote_t.h>

namespace at {
namespace native {
namespace special {
template<typename T1, typename T2>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1, T2>
exp_modified_bessel_k(T1 n, T2 x) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::exp_modified_bessel_k<T3>(n, x);
}
} // namespace special
} // namespace native
} // namespace at
