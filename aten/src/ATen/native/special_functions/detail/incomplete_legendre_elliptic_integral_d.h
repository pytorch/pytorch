#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/carlson_elliptic_r_d.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
incomplete_legendre_elliptic_integral_d(T1 k, T1 phi) {
  using T2 = numeric_t<T1>;

  if (std::isnan(k) || std::isnan(phi)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::abs(k) > T2(1)) {
    throw std::domain_error("bad `k`");
  } else {
    return std::sin(phi) * (std::sin(phi) * std::sin(phi)) * carlson_elliptic_r_d(T1(1) - std::sin(phi) * std::sin(phi),
                                                                                  T1(1) - k * k
                                                                                      * (std::sin(phi) * std::sin(phi)),
                                                                                  T1(1)) / T1(3);
  }
}
}
