#pragma once

#include <cmath>

#include <c10/util/numbers.h>
#include <ATen/native/special_functions/detail/modified_bessel.h>

namespace at::native::special_functions::detail {
template<typename T1, typename T2, typename T3>
struct spherical_modified_bessel_t {
  T1 n;
  T2 x;

  T3 i;
  T3 i_derivative;

  T3 k;
  T3 k_derivative;
};

template<typename T1>
spherical_modified_bessel_t<unsigned int, T1, T1>
spherical_modified_bessel(unsigned int n, T1 x) {
  if (std::isnan(x)) {
    const auto quiet_nan = std::numeric_limits<T1>::quiet_NaN();

    return {
        n,
        x,
        quiet_nan,
        quiet_nan,
        quiet_nan,
        quiet_nan,
    };
  } else if (x == T1(0)) {
    const auto infinity = std::numeric_limits<T1>::infinity();

    if (n == 0) {
      return {
          n,
          x,
          T1(1),
          T1(0),
          infinity,
          -infinity,
      };
    } else {
      return {
          n,
          x,
          T1(0),
          T1(0),
          infinity,
          -infinity,
      };
    }
  } else {
    const auto p = c10::numbers::sqrtpi_v<T1> / c10::numbers::sqrt2_v<T1> / std::sqrt(x);

    const auto q = modified_bessel(T1(n + 0.5L), x);

    const auto i = p * q.i;
    const auto k = p * q.k;

    return {
        n,
        x,
        i,
        p * q.i_derivative - i / (x * T1(2)),
        k,
        p * q.k_derivative - k / (x * T1(2)),
    };
  }
}
}
