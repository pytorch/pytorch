#pragma once

#include <ATen/native/special/detail/carlson_elliptic_r_d.h>
#include <ATen/native/special/detail/carlson_elliptic_r_f.h>
#include <ATen/native/special/detail/complete_legendre_elliptic_integral_e.h>
#include <ATen/native/special/detail/complex.h>
#include <ATen/native/special/detail/numeric_t.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
incomplete_legendre_elliptic_integral_e(T1 phi, T1 k) {
  using T2 = numeric_t<T1>;

  if (std::isnan(k) || std::isnan(phi) || std::abs(k) > T2(0x01)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) == 0) {
    return std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>) * carlson_elliptic_r_f(std::cos(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>) * std::cos(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>), T1(0x01) - k * k * (std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>) * std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)), T1(0x01)) - k * k * (std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>) * std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>) * std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)) * carlson_elliptic_r_d(std::cos(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>) * std::cos(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>), T1(0x01) - k * k * (std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>) * std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)), T1(0x01)) / T1(0x03);
  } else {
    return std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>) * carlson_elliptic_r_f(std::cos(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>) * std::cos(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>), T1(0x01) - k * k * (std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>) * std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)), T1(0x01)) - k * k * (std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>) * std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>) * std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)) * carlson_elliptic_r_d(std::cos(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>) * std::cos(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>), T1(0x01) - k * k * (std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>) * std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)), T1(0x01)) / T1(0x03) + T1(0x02) * std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * complete_legendre_elliptic_integral_e(k);
  }
}
} // namespace detail
} // namespace special
} // namespace native
} // namespace at
