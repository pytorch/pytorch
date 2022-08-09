#pragma once

#include <cmath>

#include <ATen/native/special_functions/detail/ln_gamma.h>
#include <ATen/native/special_functions/detail/legendre_polynomial_p.h>
#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
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
    auto P_l = legendre_polynomial_p(l, x).P_l;
    T1 fact = std::sqrt(T1(2 * l + 1) / (T1(4) * c10::numbers::pi_v<T1>));
    P_l *= fact;
    return P_l;
  } else if (x == T1(1) || x == -T1(1)) {
    return T1(0);
  } else {
    const auto sgn = (m % 2 == 1 ? -T1(1) : T1(1));
    const auto Y_mp1m_factor = x * std::sqrt(T1(2 * m + 3));
    const auto lncirc = std::log1p(-x * x);
    const auto lnpoch = ln_gamma(T1(m + 0.5L)) - ln_gamma(T1(m));
    const auto lnpre_val = -T1{0.25L} * c10::numbers::lnpi_v<T1> + T1{0.5L} * (lnpoch + m * lncirc);
    const auto sr = std::sqrt((T1(2) + T1(1) / m) / (T1(4) * c10::numbers::pi_v<T1>));
    auto Y_mm = sgn * sr * std::exp(lnpre_val);
    auto Y_mp1m = Y_mp1m_factor * Y_mm;

    if (l == m) {
      return Y_mm;
    } else if (l == m + 1) {
      return Y_mp1m;
    } else {
      auto Y_lm = T1(0);

      for (auto ll = m + 2; ll <= l; ll++) {
        const auto rat1 = T1(ll - m) / T1(ll + m);
        const auto rat2 = T1(ll - m - 1) / T1(ll + m - 1);
        const auto fact1 = std::sqrt(rat1 * T1(2 * ll + 1) * T1(2 * ll - 1));
        const auto fact2 = std::sqrt(rat1 * rat2 * T1(2 * ll + 1) / T1(2 * ll - 3));
        Y_lm = (x * Y_mp1m * fact1 - T1(ll + m - 1) * Y_mm * fact2) / T1(ll - m);
        Y_mm = Y_mp1m;
        Y_mp1m = Y_lm;
      }

      return Y_lm;
    }
  }
}
}
