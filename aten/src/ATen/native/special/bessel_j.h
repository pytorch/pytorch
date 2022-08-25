#pragma once

#include <ATen/native/special/detail/bessel.h>
#include <ATen/native/special/detail/ln_gamma.h>
#include <ATen/native/special/detail/numeric_t.h>
#include <ATen/native/special/detail/promote_t.h>

namespace at {
namespace native {
namespace special {
template<typename T1, typename T2>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1, T2>
bessel_j(T1 v, T2 z) {
  using T3 = detail::promote_t<T1, T2>;
  using T4 = detail::numeric_t<T3>;

  if (std::isnan(v) || std::isnan(z) || z < T3(0)) {
    return std::numeric_limits<T3>::quiet_NaN();
  } else if (v >= T3(0) && z * z < T3(10) * (v + T3(1))) {
    if (std::abs(z) < std::numeric_limits<T4>::epsilon()) {
      if (v == T3(0)) {
        return T3(1);
      } else {
        return T3(0);
      }
    } else {
      auto p = T3(1);
      auto q = T3(1);

      for (auto j = 1; j < 200; j++) {
        q = q * (T3(-1) * (z / T4(2)) * (z / T4(2)) / (T3(j) * (T3(v) + T3(j))));
        p = p + q;

        if (std::abs(q / p) < std::numeric_limits<T4>::epsilon()) {
          break;
        }
      }

      return std::exp(T3(v) * std::log(z / T4(2)) - detail::ln_gamma(T4(1) + v)) * p;
    }
  } else {
    return detail::bessel(v, z).j;
  }
}
} // namespace special
} // namespace native
} // namespace at
