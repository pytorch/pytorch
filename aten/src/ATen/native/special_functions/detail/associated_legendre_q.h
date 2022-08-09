#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>

namespace at::native::special_functions::detail {
template<typename Tp>
struct associated_legendre_q_t {
  unsigned int l; /// degree
  unsigned int m; /// order
  Tp x; /// argument
  Tp Q_lm;   /// Q_l^{(m)}(x)
  Tp Q_lmm1; /// Q_l^{(m-1)}(x)
  Tp Q_lmm2; /// Q_l^{(m-2)}(x)
  Tp phase = 1; // -1 For Condon-Shortley.

  constexpr Tp
  deriv() const noexcept {
    if (std::abs(x) == 1)
      return Tp(l % 2 == 1 ? -1 : +1)
          * std::numeric_limits<Tp>::infinity();
    else {
      const auto fact = (Tp{1} - x) * (Tp{1} + x);
      const auto root = std::sqrt(Tp{1} - x)
          * std::sqrt(Tp{1} + x);
      return Tp(m) * x * Q_lm / fact
          + Tp(l + m) * Tp(l - m + 1) * Q_lmm1 / root;
    }
  }
};

template<typename T1>
associated_legendre_q_t<T1>
associated_legendre_q(unsigned int l, unsigned int m, T1 x, T1 phase = T1(1)) {
  using T2 = numeric_t<T1>;
  if (std::isnan(x)) {
    return {
        l,
        m,
        x,
        std::numeric_limits<T2>::quiet_NaN(),
        std::numeric_limits<T2>::quiet_NaN(),
        std::numeric_limits<T2>::quiet_NaN(),
        phase,
    };
  } else if (std::abs(x) < T2(1)) {
    if (l == 0) {
      if (m == 0) {
        return {l, m, x, std::log((T1(1) + x) / (T1(1) - x)) / T1(2), T1(0), T1(0), phase};
      } else if (m == 1) {
        return {l, m, x, phase / (std::sqrt(T1(1) - x) * std::sqrt(T1(1) + x)),
                std::log((T1(1) + x) / (T1(1) - x)) / T1(2), T1(0), phase};
      }
    }

    if (l == 1) {
      if (m == 0) {
        return {l, m, x, x * (std::log((T1(1) + x) / (T1(1) - x)) / T1(2)) - T1(1), T1(0), T1(0), phase};
      } else if (m == 1) {
        return {l, m, x, phase * (std::sqrt(T1(1) - x) * std::sqrt(T1(1) + x))
            * (std::log((T1(1) + x) / (T1(1) - x)) / T1(2) + x / ((T1(1) - x) * (T1(1) + x))),
                x * (std::log((T1(1) + x) / (T1(1) - x)) / T1(2)) - T1(1), T1(0), phase};
      }
    }

    auto p = std::log((T1(1) + x) / (T1(1) - x)) / T1(2);
    auto q = x * (std::log((T1(1) + x) / (T1(1) - x)) / T1(2)) - T1(1);
    auto r = phase / (std::sqrt(T1(1) - x) * std::sqrt(T1(1) + x));
    auto s = phase * (std::sqrt(T1(1) - x) * std::sqrt(T1(1) + x)) * (
        std::log((T1(1) + x) / (T1(1) - x)) / T1(2) + x / ((T1(1) - x) * (T1(1) + x)));
    auto t = (T1(3) * x * q - T1(1) * p) / T1(2);
    auto u = (T1(3) * x * s - T1(2) * r);

    for (unsigned int j = 3; j <= l; j++) {
      p = q;
      q = t;
      t = (T1(2 * j - 1) * x * q - T1(j - 1) * p) / T1(j);
      r = s;
      s = u;
      u = (T1(2 * j - 1) * x * s - T1(j) * r) / T1(j - 1);
    }

    if (m == 0) {
      return {l, m, x, t, T1(0), T1(0), phase};
    } else if (m == 1) {
      return {l, m, x, u, t, T1(0), phase};
    }

    auto a = t;
    auto b = u;
    auto c = phase * (T1(2) * x * b / (std::sqrt(T1(1) - x) * std::sqrt(T1(1) + x)) + T1(l + 1) * T1(l) * t);

    for (unsigned int k = 3; k <= m; k++) {
      a = b;
      b = c;
      c = phase * (T1(2 * (k - 1)) * x * b / (std::sqrt(T1(1) - x) * std::sqrt(T1(1) + x))
          - T1(l + k - 1) * T1(l - k + 2) * t);
    }

    return {
        l,
        m,
        x,
        c,
        b,
        a,
        phase,
    };
  } else {
    return {
        l,
        m,
        x,
        T1(0),
        T1(0),
        T1(0),
        phase,
    };
  }
}
}
