#pragma once

#include <cmath>

#include <ATen/native/special/detail/ln_gamma.h>
#include <ATen/native/special/detail/legendre_polynomial_p.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
spherical_legendre_y(unsigned int l, unsigned int m, T1 theta) {
  if (std::isnan(theta)) {
    return std::numeric_limits<T1>::quiet_NaN();
  }

  const auto x = std::cos(theta);

  if (m > l) {
    return T1(0);
  } else if (m == 0) {
    return legendre_polynomial_p(l, x).p_n * std::sqrt(T1(2 * l + 1) / (T1(4) * c10::numbers::pi_v<T1>));
  } else if (x == T1(1) || x == -T1(1)) {
    return T1(0);
  } else {
    auto q = (m % 2 == 1 ? -T1(1) : T1(1)) * std::sqrt((T1(2) + T1(1) / m) / (T1(4) * c10::numbers::pi_v<T1>)) * std::exp(-T1{0.25L} * c10::numbers::lnpi_v<T1> + T1{0.5L} * (ln_gamma(T1(m + 0.5L)) - ln_gamma(T1(m)) + m * std::log1p(-x * x)));
    auto r = x * std::sqrt(T1(2 * m + 3)) * q;

    if (l == m) {
      return q;
    } else if (l == m + 1) {
      return r;
    } else {
      auto p = T1(0);

      for (auto j = m + 2; j <= l; j++) {
        p = (x * r * std::sqrt(T1(j - m) / T1(j + m) * T1(2 * j + 1) * T1(2 * j - 1)) - T1(j + m - 1) * q * std::sqrt(T1(j - m) / T1(j + m) * T1(j - m - 1) / T1(j + m - 1) * T1(2 * j + 1) / T1(2 * j - 3))) / T1(j - m);
        q = r;
        r = p;
      }

      return p;
    }
  }
}
}
}
}
}
