#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>

namespace at::native::special_functions::detail {
template<typename Tp>
struct legendre_polynomial_p_t {
  unsigned int l;
  Tp x;
  Tp P_l;   /// P_l(x)
  Tp P_lm1; /// P_{l-1}(x)
  Tp P_lm2; /// P_{l-2}(x)

  // Return the Lobatto polynomial.
  constexpr Tp
  lobatto() const noexcept { return l * (P_l - x * P_lm1); }

  constexpr Tp
  deriv() const noexcept {
    if (std::abs(x) == Tp{1}) {
      const auto sgn = x == Tp{+1}
                       ? Tp{+1}
                       : (l % 2 == 0 ? Tp{-1} : Tp{+1});
      return sgn * Tp(l) * Tp(l + 1) / Tp{2};
    } else
      return l * (x * P_l - P_lm1)
          / ((Tp{1} - x) * (Tp{1} + x));
  }
};

template<typename T1>
legendre_polynomial_p_t<T1>
legendre_polynomial_p(unsigned int l, T1 x) {
  using T2 = numeric_t<T1>;
  using T3 = legendre_polynomial_p_t<T1>;

  if (std::isnan(x)) {
    return {l, std::numeric_limits<T2>::quiet_NaN(), std::numeric_limits<T2>::quiet_NaN(),
            std::numeric_limits<T2>::quiet_NaN(), std::numeric_limits<T2>::quiet_NaN()};
  } else if (x == T2{+1}) {
    return {l, x, T1{+1}, l >= 1 ? T1{+1} : T1(0), l >= 2 ? T1{+1} : T1(0)};
  } else if (x == T2(-1)) {
    if (l % 2 == 1) {
      return T3{l, x, T1(-1), +(l >= 1 ? T1{+1} : T1(0)), -(l >= 2 ? T1{+1} : T1(0))};
    } else {
      return T3{l, x, T1{+1}, -(l >= 1 ? T1{+1} : T1(0)), +(l >= 2 ? T1{+1} : T1(0))};
    }
  } else if (l == 0) {
    return {l, x, T1(1), T1(0), T1(0)};
  } else if (l == 1) {
    return {l, x, x, T1(1), T1(0)};
  } else {
    auto p = x;
    auto q = T1(1);
    auto r = T1(2) * x * p - q - (x * p - q) / T1(2);

    for (unsigned int j = 3; j <= l; j++) {
      q = p;
      p = r;
      r = T1(2) * x * p - q - (x * p - q) / T1(j);
    }

    return {l, x, r, p, q};
  }
}
}
