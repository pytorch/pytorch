#pragma once

#include <ATen/native/special_functions/detail/incomplete_elliptic_integral_f.h>
#include <ATen/native/special_functions/detail/carlson_elliptic_r_f.h>
#include <ATen/native/special_functions/detail/carlson_elliptic_r_j.h>
#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/jacobi_zeta.h>
#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
heuman_lambda(T1 k, T1 phi) {
  using T2 = numeric_t<T1>;

  if (std::isnan(k) || std::isnan(phi)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::abs(k) > T1(1)) {
    throw std::domain_error("heuman_lambda: bad argument");
  } else if (std::abs(std::abs(k) - T1(1)) < std::numeric_limits<T1>::epsilon()) {
    return phi / (c10::numbers::pi_v<T1> / T1(2));
  } else if (std::abs(k) < std::numeric_limits<T1>::epsilon()) {
    return std::sin(phi);
  } else if (std::real(phi) < T2(0)) {
    return -heuman_lambda(k, -phi);
  } else if (std::abs(phi - c10::numbers::pi_v<T1> / T1(2)) < T1(5) * std::numeric_limits<T1>::epsilon()) {
    return T1(1);
  } else if (std::abs(phi) < T1(5) * std::numeric_limits<T1>::epsilon()) {
    return T1(0);
  } else if (std::abs(phi) < c10::numbers::pi_v<T1> / T1(2)) {
    if (std::abs(T1(1) - (T1(1) - k * k) * std::sin(phi) * std::sin(phi)) < T2(0)) {
      return T1(2) * (T1(1) - k * k) * std::cos(phi) * std::sin(phi) / (c10::numbers::pi_v<T1> * std::sqrt(T1(0)))
          * (carlson_elliptic_r_f(T1(0), T1(1) - k * k, T1(1))
              + k * k / (T1(3) * T1(0)) * carlson_elliptic_r_j(T1(0), T1(1) - k * k, T1(1), T1(1) - k * k / T1(0)));
    } else {
      return T1(2) * (T1(1) - k * k) * std::cos(phi) * std::sin(phi)
          / (c10::numbers::pi_v<T1> * std::sqrt(T1(1) - (T1(1) - k * k) * std::sin(phi) * std::sin(phi)))
          * (carlson_elliptic_r_f(T1(0), T1(1) - k * k, T1(1))
              + k * k / (T1(3) * (T1(1) - (T1(1) - k * k) * std::sin(phi) * std::sin(phi)))
                  * carlson_elliptic_r_j(T1(0),
                                         T1(1) - k * k,
                                         T1(1),
                                         T1(1) - k * k / (T1(1) - (T1(1) - k * k) * std::sin(phi) * std::sin(phi))));
    }
  } else {
    return incomplete_elliptic_integral_f(std::sqrt(T1(1) - k * k), phi)
        / complete_elliptic_integral_k(std::sqrt(T1(1) - k * k))
        + complete_elliptic_integral_k(k) * jacobi_zeta(std::sqrt(T1(1) - k * k), phi)
            / (c10::numbers::pi_v<T1> / T1(2));
  }
}
}
