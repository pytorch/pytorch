#pragma once

#include <ATen/native/special/detail/carlson_elliptic_r_f.h>
#include <ATen/native/special/detail/carlson_elliptic_r_j.h>
#include <ATen/native/special/detail/complex.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
bulirsch_elliptic_integral_cel(T1 k_c, T1 p, T1 a, T1 b) {
  if (std::isnan(k_c) || std::isnan(p) || std::isnan(a) || std::isnan(b)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    return a * carlson_elliptic_r_f(T1(0x00), k_c * k_c, T1(0x01)) + (b - p * a) * carlson_elliptic_r_j(T1(0x00), k_c * k_c, T1(0x01), p) / T1(0x03);
  }
}
} // namespace detail
} // namespace special
} // namespace native
} // namespace at
