#pragma once

#include <ATen/native/special/detail/bessel.h>
#include <ATen/native/special/detail/promote_t.h>

namespace at {
namespace native {
namespace special {
template<typename T1, typename T2>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1, T2>
bessel_y(T1 v, T2 z) {
  using T3 = detail::promote_t<T1, T2>;

  if (std::isnan(v) || std::isnan(z) || z < T3(0)) {
    return std::numeric_limits<T3>::quiet_NaN();
  } else {
    return detail::bessel(v, z).y;
  }
}
} // namespace special
} // namespace native
} // namespace at
