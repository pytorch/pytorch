#pragma once

#include <complex>

#include <ATen/native/special_functions/detail/spherical_legendre_y.h>

namespace at::native::special_functions::detail {
template<typename T1>
std::complex<T1>
spherical_harmonic_y(unsigned int l, int m, T1 theta, T1 phi) {
  if (std::isnan(theta) || std::isnan(phi)) {
    return std::complex<T1>{std::numeric_limits<T1>::quiet_NaN(), std::numeric_limits<T1>::quiet_NaN()};
  } else if (std::abs(m) > l) {
    return std::complex<T1>{0, 0};
  } else {
    return spherical_legendre_y(l, std::abs(m), theta) * std::polar(T1(1), T1(m) * phi);
  }
}
}
