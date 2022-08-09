#pragma once

#include <ATen/native/special_functions/detail/complete_elliptic_integral_e.h>
#include <ATen/native/special_functions/detail/carlson_elliptic_r_d.h>
#include <ATen/native/special_functions/detail/carlson_elliptic_r_f.h>
#include <ATen/native/special_functions/detail/numeric_t.h>
#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
incomplete_elliptic_integral_e(T1 phi, T1 k) {
  using T2 = numeric_t<T1>;

  if (std::isnan(k) || std::isnan(phi)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::abs(k) > T2(1)) {
    throw std::domain_error("bad `k`");
  } else if (std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) == 0) {
    return std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)
        * carlson_elliptic_r_f(
            std::cos(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)
                * std::cos(
                    phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>),
            T1(1) - k * k * (std::sin(
                phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)
                * std::sin(
                    phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)),
            T1(1)) - k * k
        * (std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)
            * std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)
            * std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>))
        * carlson_elliptic_r_d(
            std::cos(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)
                * std::cos(
                    phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>),
            T1(1) - k * k * (std::sin(
                phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)
                * std::sin(
                    phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)),
            T1(1)) / T1(3);
  } else {
    return std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)
        * carlson_elliptic_r_f(
            std::cos(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)
                * std::cos(
                    phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>),
            T1(1) - k * k * (std::sin(
                phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)
                * std::sin(
                    phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)),
            T1(1)) - k * k
        * (std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)
            * std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)
            * std::sin(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>))
        * carlson_elliptic_r_d(
            std::cos(phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)
                * std::cos(
                    phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>),
            T1(1) - k * k * (std::sin(
                phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)
                * std::sin(
                    phi - std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * c10::numbers::pi_v<T2>)),
            T1(1)) / T1(3)
        + T1(2) * std::floor(std::real(phi) / c10::numbers::pi_v<T2> + T2(0.5L)) * complete_elliptic_integral_e(k);
  }
}
}
