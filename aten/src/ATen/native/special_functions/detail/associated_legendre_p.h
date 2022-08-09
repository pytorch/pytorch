#pragma once

#include "legendre_polynomial_p.h"
#include <ATen/native/special_functions/detail/numeric_t.h>

namespace at::native::special_functions::detail {
template<typename T1>
struct associated_legendre_p_t {
  unsigned int l;
  unsigned int m;
  T1 x;
  T1 P_lm;   /// P_l^{(m)}(x)
  T1 P_lm1m; /// P_{l-1}^{(m)}(x)
  T1 P_lm2m; /// P_{l-2}^{(m)}(x)
  T1 phase = 1; // -1 For Condon-Shortley.

  constexpr T1
  deriv() const noexcept {
    if (std::abs(x) == T1(1)) {
      if (m == 0) {
        return (x == T1(1) ? T1(1) : (l % 2 == 0 ? T1(-1) : T1(1))) * T1(l) * T1(l + 1) / T1(2);
      } else if (m == 1) {
        return -phase * (x == T1(1) ? T1(1) : (l % 2 == 0 ? T1(1) : T1(-1))) * std::numeric_limits<T1>::infinity();
      } else if (m == 2) {
        return -(x == T1(1) ? T1(1) : (l % 2 == 0 ? T1(-1) : T1(1))) * T1(l + 2) * T1(l + 1) / T1(2) * T1(l)
            * T1(int(l) - 1) / T1(2);
      } else {
        return T1(0);
      }
    } else {
      return -phase * ((l + m) * P_lm1m - l * x * P_lm) / ((T1(1) - x) * (T1(1) + x));
    }
  }
};

template<typename T1>
associated_legendre_p_t<T1>
associated_legendre_p(unsigned int l, unsigned int m, T1 x, T1 phase = T1(1)) {
  using T2 = numeric_t<T1>;
  if (m > l) {
    return {l, m, x, T1(0), T1(0), T1(0)};
  } else if (std::isnan(x)) {
    return {l, m, x, std::numeric_limits<T2>::quiet_NaN(), std::numeric_limits<T2>::quiet_NaN(),
            std::numeric_limits<T2>::quiet_NaN(), phase};
  } else if (m == 0) {
    const auto P_l = legendre_polynomial_p(l, x);

    return {l, m, x, P_l.P_l, P_l.P_lm1, P_l.P_lm2, phase};
  } else {
    auto P_mm = T1(1);

    if (m > 0) {
      const auto root = std::sqrt(T1(1) - x) * std::sqrt(T1(1) + x);
      auto fact = T1(1);

      for (unsigned int i = 1; i <= m; i++) {
        P_mm = P_mm * (phase * fact * root);
        fact = fact + T1(2);
      }
    }

    if (l == m) {
      return {l, m, x, P_mm, T1(0), T1(0), phase};
    }

    auto P_mp1m = T1(2 * m + 1) * x * P_mm;

    if (l == m + 1) {
      return {l, m, x, P_mp1m, P_mm, T1(0), phase};
    }

    auto a = P_mm;
    auto b = P_mp1m;
    auto c = (T1(2 * m + 3) * x * b - T1(2 * m + 1) * a) / T1(2);

    for (unsigned int k = m + 3; k <= l; k++) {
      a = b;
      b = c;
      c = (T1(2 * k - 1) * x * b - T1(k + m - 1) * a) / T1(k - m);
    }

    return {l, m, x, c, b, a};
  }
}
}
