#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/carlson_elliptic_r_j.h>
#include <ATen/native/special_functions/detail/complete_elliptic_integral_k.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
jacobi_zeta(T1 k, T1 phi) {
  using T2 = numeric_t<T1>;

  if (std::isnan(k) || std::isnan(phi)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::abs(k) > T2(1)) {
    throw std::domain_error("jacobi_zeta: bad argument");
  } else if (std::abs(k) < std::numeric_limits<T1>::epsilon()) {
    return T1(0);
  } else if (std::abs(k - T2(1)) < std::numeric_limits<T1>::epsilon()) {
    return std::sin(phi);
  } else if (std::real(phi) < T2(0)) {
    return -jacobi_zeta(k, -phi);
  } else if (std::abs(phi) < std::numeric_limits<T1>::epsilon()
      || std::abs(phi - c10::numbers::pi_v<T1> / T1(2)) < std::numeric_limits<T1>::epsilon()) {
    return T1(0);
  } else {
    return k * k * std::cos(phi) * std::sin(phi) * std::sqrt(T1(1) - k * k * std::sin(phi) * std::sin(phi))
        * carlson_elliptic_r_j(T1(0), T1(1) - k * k, T2(1), T1(1) - k * k * std::sin(phi) * std::sin(phi))
        / (T1(3) * complete_elliptic_integral_k(k));
  }
}
}
}
}
}
