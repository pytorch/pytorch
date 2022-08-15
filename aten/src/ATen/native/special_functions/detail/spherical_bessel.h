#pragma once

#include <cmath>
#include <complex>

#include <ATen/native/special_functions/detail/bessel.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
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

  return {n, x, (c10::numbers::sqrtpi_v<T1> / c10::numbers::sqrt2_v<T1>) / std::sqrt(x) * bessel(x, T1(n + 0.5L)).j,
          (c10::numbers::sqrtpi_v<T1> / c10::numbers::sqrt2_v<T1>) / std::sqrt(x) * bessel(x, T1(n + 0.5L)).j_derivative
              - (c10::numbers::sqrtpi_v<T1> / c10::numbers::sqrt2_v<T1>) / std::sqrt(x) * bessel(x, T1(n + 0.5L)).j
                  / (T1(2) * x),
          (c10::numbers::sqrtpi_v<T1> / c10::numbers::sqrt2_v<T1>) / std::sqrt(x) * bessel(x, T1(n + 0.5L)).y,
          (c10::numbers::sqrtpi_v<T1> / c10::numbers::sqrt2_v<T1>) / std::sqrt(x) * bessel(x, T1(n + 0.5L)).y_derivative
              - (c10::numbers::sqrtpi_v<T1> / c10::numbers::sqrt2_v<T1>) / std::sqrt(x) * bessel(x, T1(n + 0.5L)).y
                  / (T1(2) * x),};
}

template<typename T1>
spherical_bessel_t<unsigned int, T1, std::complex<T1>>
spherical_bessel_negative_x(unsigned int n, T1 x) {
  using T2 = std::complex<T1>;
  using T3 = spherical_bessel_t<unsigned int, T1, T2>;

  if (x >= T1(0)) {
    throw std::domain_error("non-negative `x`");
  } else {
    return {n, x, (c10::numbers::sqrtpi_v<T1> / c10::numbers::sqrt2_v<T1>) / std::sqrt(T2(x))
        * bessel_negative_x(T1(n + 0.5L), x).j,
            (c10::numbers::sqrtpi_v<T1> / c10::numbers::sqrt2_v<T1>) / std::sqrt(T2(x))
                * bessel_negative_x(T1(n + 0.5L), x).j_derivative
                - (c10::numbers::sqrtpi_v<T1> / c10::numbers::sqrt2_v<T1>) / std::sqrt(T2(x))
                    * bessel_negative_x(T1(n + 0.5L), x).j / (T1(2) * x),
            (c10::numbers::sqrtpi_v<T1> / c10::numbers::sqrt2_v<T1>) / std::sqrt(T2(x))
                * bessel_negative_x(T1(n + 0.5L), x).y,
            (c10::numbers::sqrtpi_v<T1> / c10::numbers::sqrt2_v<T1>) / std::sqrt(T2(x))
                * bessel_negative_x(T1(n + 0.5L), x).y_derivative
                - (c10::numbers::sqrtpi_v<T1> / c10::numbers::sqrt2_v<T1>) / std::sqrt(T2(x))
                    * bessel_negative_x(T1(n + 0.5L), x).y / (T1(2) * x)};
  }
}
}
}
}
}
