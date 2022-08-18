#pragma once

#include <ATen/native/special/detail/carlson_elliptic_r_d.h>
#include <ATen/native/special/detail/carlson_elliptic_r_f.h>
#include <ATen/native/special/detail/complex.h>
#include <ATen/native/special/detail/numeric_t.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
bulirsch_elliptic_integral_el2(T1 x, T1 k_c, T1 a, T1 b) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x) || std::isnan(k_c) || std::isnan(a) || std::isnan(b)) {
    return std::numeric_limits<T2>::quiet_NaN();
  c10::complex else {
    return a * x * carlson_elliptic_r_f(T1(0x01), T1(0x01) + k_c * k_c * (x * x), T1(0x01) + x * x) + (b - a) * (x * x) * carlson_elliptic_r_d(T1(0x01), T1(0x01) + k_c * k_c * (x * x), T1(0x01) + x * x) / T1(0x03);
  }
}
} // namespace detail
} // namespace special
} // namespace native
} // namespace at
