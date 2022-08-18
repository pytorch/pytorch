#pragma once

#include <ATen/native/special/detail/complete_carlson_elliptic_r_f.h>
#include <ATen/native/special/detail/complex.h>
#include <ATen/native/special/detail/numeric_t.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
complete_legendre_elliptic_integral_k(T1 k) {
  using T2 = numeric_t<T1>;

  if (std::isnan(k) || std::abs(k) == T2(0x01)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    return complete_carlson_elliptic_r_f(T1(0x01) - k * k, T1(0x01));
  }
}
} // namespace detail
} // namespace special
} // namespace native
} // namespace at
