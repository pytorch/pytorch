#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/gauss_hypergeometric_2_f_1.h>

namespace at::native::special_functions::detail {
template<typename Tp>
struct legendre_q_t {
  unsigned int l;
  Tp x;
  Tp Q_l;   /// Q_l(x)
  Tp Q_lm1; /// Q_{l-1}(x)
  Tp Q_lm2; /// Q_{l-2}(x)

  constexpr Tp
  deriv() const noexcept {
    if (std::abs(x) == Tp{1})
      return Tp(l % 2 == 1 ? -1 : +1)
          * std::numeric_limits<Tp>::infinity();
    else
      return Tp(l) * (x * Q_l - Q_lm1)
          / ((Tp{1} - x) * (Tp{1} + x));
  }
};

template<typename T2>
T2
legendre_q_series(unsigned int l, T2 x) {
  auto p = T2(1) / x;

  for (unsigned int k = 1; k <= l; k++) {
    p = p * (T2(k) * (T2(1) / x) / T2(2 * k - 1));
  }

  return p * at::native::special_functions::gauss_hypergeometric_2_f_1(T2(l + 1) / T2(2),
                                                                         T2(l + 2) / T2(2),
                                                                         T2(2 * l + 3) / T2(2),
                                                                         T2(1) / x * (T2(1) / x));
}

template<typename T1>
legendre_q_t<T1>
legendre_q(unsigned int l, T1 x) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x)) {
    return {l, x, std::numeric_limits<T2>::quiet_NaN(), std::numeric_limits<T2>::quiet_NaN(),
            std::numeric_limits<T2>::quiet_NaN()};
  } else if (std::abs(x - T2(1)) < std::numeric_limits<T2>::epsilon()) {
    return {l, x, std::numeric_limits<T2>::infinity(), std::numeric_limits<T2>::infinity(),
            std::numeric_limits<T2>::infinity()};
  } else if (std::abs(x + T2(1)) < std::numeric_limits<T2>::epsilon()) {
    return {l, x, (l & 1 ? +1 : -1) * std::numeric_limits<T2>::infinity(),
            -(l & 1 ? +1 : -1) * std::numeric_limits<T2>::infinity(),
            (l & 1 ? +1 : -1) * std::numeric_limits<T2>::infinity()};
  } else if (std::abs(x) < T2(1)) {
    if (l == 0) {
      return {l, x, T1{0.5L} * std::log((T1(1) + x) / (T1(1) - x)), T1(0), T1(0)};
    } else if (l == 1) {
      return {l, x, x * (T1{0.5L} * std::log((T1(1) + x) / (T1(1) - x))) - T1(1),
              T1{0.5L} * std::log((T1(1) + x) / (T1(1) - x)), T1(0)};
    } else {
      auto p = x * (T1{0.5L} * std::log((T1(1) + x) / (T1(1) - x))) - T1(1);
      auto q = T1{0.5L} * std::log((T1(1) + x) / (T1(1) - x));
      auto r = T1(2) * x * p - q - (x * p - q) / T1(2);

      for (unsigned int ll = 3; ll <= l; ++ll) {
        q = p;
        p = r;
        r = T1(2) * x * p - q - (x * p - q) / T1(ll);
      }

      return {l, x, r, p, q};
    }
  } else {
    if (l == 0) {
      return {l, x, legendre_q_series(l, x), T1(0), T1(0)};
    } else {
      if (l == 1) {
        return {l, x, legendre_q_series(l, x), legendre_q_series(l - 1, x), T1(0)};
      } else {
        return {l, x, legendre_q_series(l, x), legendre_q_series(l - 1, x),
                (T1(2 * l - 1) * x * legendre_q_series(l - 1, x) - T1(l) * legendre_q_series(l, x)) / T1(l - 1)};
      }
    }
  }
}
}
