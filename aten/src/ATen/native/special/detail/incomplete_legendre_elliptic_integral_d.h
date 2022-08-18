#pragma once

#include <ATen/native/special/detail/carlson_elliptic_r_d.h>
#include <ATen/native/special/detail/complex.h>
#include <ATen/native/special/detail/numeric_t.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
incomplete_legendre_elliptic_integral_d(T1 k, T1 phi) {
  using T2 = numeric_t<T1>;

  if (std::isnan(k) || std::isnan(phi) || std::abs(k) > T2(0x01)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    return std::sin(phi) * (std::sin(phi) * std::sin(phi)) * carlson_elliptic_r_d(T1(0x01) - std::sin(phi) * std::sin(phi), T1(0x01) - k * k * (std::sin(phi) * std::sin(phi)), T1(0x01)) / T1(0x03);
  }
}
} // namespace detail
} // namespace special
} // namespace native
} // namespace at
