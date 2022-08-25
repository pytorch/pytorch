#pragma once

#include <cmath>
#include <complex>

#include <ATen/native/special/detail/bessel.h>
#include <ATen/native/special/detail/bessel_negative_z.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1, typename T2, typename T3>
struct spherical_bessel_t {
  T1 n;
  T2 x;

  T3 j;
  T3 j_derivative;

  T3 y;
  T3 y_derivative;
};

template<typename T1>
spherical_bessel_t<unsigned int, T1, T1>
spherical_bessel(unsigned int n, T1 x) {
  using T2 = spherical_bessel_t<unsigned int, T1, T1>;
  
  const auto p = bessel(T1(n + 0.5L), x);
  const auto q = c10::numbers::sqrtpi_v<T1> / c10::numbers::sqrt2_v<T1> / std::sqrt(x);
  const auto r = x * T1(2);

  const auto j = q * p.j;
  const auto y = q * p.y;

  const auto j_derivative = q * p.j_derivative - j / r;
  const auto y_derivative = q * p.y_derivative - y / r;

  return {
      n,
      x,
      j,
      j_derivative,
      y,
      y_derivative,
  };
}

template<typename T1>
spherical_bessel_t<unsigned int, T1, c10::complex<T1>>
spherical_bessel_negative_x(unsigned int n, T1 x) {
  using T2 = c10::complex<T1>;
  using T3 = spherical_bessel_t<unsigned int, T1, T2>;

  if (x >= T1(0)) {
    throw std::domain_error("non-negative `x`");
  } else {
    const auto p = bessel_negative_z(T1(n + 0.5L), x);
    const auto q = c10::numbers::sqrtpi_v<T1> / c10::numbers::sqrt2_v<T1> / std::sqrt(T2(x));
    const auto r = x * T1(2);

    const auto j = q * p.j;
    const auto y = q * p.y;

    const auto j_derivative = q * p.j_derivative - j / r;
    const auto y_derivative = q * p.y_derivative - y / r;

    return {
        n,
        x,
        j,
        j_derivative,
        y,
        y_derivative,
    };
  }
}
}
}
}
}
