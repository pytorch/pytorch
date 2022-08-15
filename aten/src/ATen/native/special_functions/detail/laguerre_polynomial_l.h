#pragma once

#include <cmath>

#include <c10/util/numbers.h>
#include <ATen/native/special_functions/detail/ln_gamma.h>
#include <ATen/native/special_functions/sin_pi.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename Tpa, typename Tp>
struct laguerre_t {
  unsigned int n;
  Tpa alpha1;
  Tp x;
  Tp L_n;
  Tp L_nm1;
  Tp L_nm2;

  constexpr Tp
  deriv() const noexcept { return (Tp(n) * L_nm1 - Tp(n + alpha1) * L_nm2) / x; }
};

template<typename T1, typename T2>
T2
laguerre_large_n(unsigned n, T1 alpha1, T2 x) {
  return std::exp(ln_gamma(T2(n) + (T2(alpha1) + T2(1))) - ln_gamma(T2(n + 1)) + T2{0.5L} * x
                      + T2{0.5L} * (T2(1) - (T2(alpha1) + T2(1)))
                          * std::log(T2{0.25L} * x * (T2(2) * (T2(alpha1) + T2(1)) - T2(4) * -T2(n))) - T2{0.25L}
      * std::log((c10::numbers::pi_v<T2> / T2(2)) * (c10::numbers::pi_v<T2> / T2(2))
                     * (T2(2) * (T2(alpha1) + T2(1)) - T2(4) * -T2(n)) * (T2(2) * (T2(alpha1) + T2(1)) - T2(4) * -T2(n))
                     * (x / (T2(2) * (T2(alpha1) + T2(1)) - T2(4) * -T2(n)))
                     * (T2(1) - x / (T2(2) * (T2(alpha1) + T2(1)) - T2(4) * -T2(n)))))
      * (at::native::special_functions::sin_pi(-T2(n)) + (std::sin(
          T2{0.25L} * (T2(2) * (T2(alpha1) + T2(1)) - T2(4) * -T2(n))
              * (T2(2) * std::acos(std::sqrt(x / (T2(2) * (T2(alpha1) + T2(1)) - T2(4) * -T2(n))))
                  - std::sin(T2(2) * std::acos(std::sqrt(x / (T2(2) * (T2(alpha1) + T2(1)) - T2(4) * -T2(n))))))
              + c10::numbers::pi_v<T2> / T2(4))));
}

template<typename T1, typename T2>
laguerre_t<T1, T2>
laguerre_recurrence(unsigned int n, T1 alpha1, T2 x) {
  if (n == 0) {
    return {n, alpha1, x, T2(1), T2(0), T2(0)};
  } else if (n == 1) {
    return {n, alpha1, x, -x + T2(1) + T2(alpha1), T2(1), T2(0)};
  } else {
    auto q = T2(1);
    auto p = -x + T2(1) + T2(alpha1);
    auto r = (T2(3) + T2(alpha1) - x) * p / T2(2) - (T2(1) + T2(alpha1)) * q / T2(2);

    for (unsigned int nn = 3; nn <= n; ++nn) {
      q = p;
      p = r;
      r = (T2(2 * nn - 1) + T2(alpha1) - x) * p / T2(nn) - (T2(nn - 1) + T2(alpha1)) * q / T2(nn);
    }

    return {n, alpha1, x, r, p, q};
  }
}

template<typename Tpa, typename Tp>
Tp
laguerre_hyperg(unsigned int n, Tpa alpha1, Tp x) {
  auto tc = Tp{1};
  const auto ax = std::abs(x);
  for (unsigned int k = 1; k <= n; ++k)
    tc *= (ax / k);

  auto term = tc * (x < Tp{0} ? Tp{1} : ((n % 2 == 1) ? -Tp{1} : Tp{1}));
  auto sum = term;
  for (int k = int(n) - 1; k >= 0; --k) {
    term *= ((Tp(alpha1) + Tp{1} + Tp(k)) / Tp(int(n) - k)) * Tp(k + 1) / -x;
    sum += term;
  }

  return sum;
}

template<typename T1, typename T2>
T2
laguerre_polynomial_l(unsigned int n, T1 alpha1, T2 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else if (n == 0) {
    return T2(1);
  } else if (n == 1) {
    return T2(1) + T2(alpha1) - x;
  } else if (x == T2(0)) {
    auto product = T2(alpha1) + T2(1);

    for (unsigned int k = 2; k <= n; k++) {
      product = product * ((T2(alpha1) + T2(k)) / T2(k));
    }

    return product;
  } else if (n > 10000000 && T2(alpha1) > -T2(1) && x < T2(2) * (T2(alpha1) + T2(1)) + T2(4 * n)) {
    return laguerre_large_n(n, alpha1, x);
  } else if (T2(alpha1) >= T2(0) || (x > T2(0) && T2(alpha1) < -T2(n + 1))) {
    return laguerre_recurrence(n, alpha1, x).L_n;
  } else {
    return laguerre_hyperg(n, alpha1, x);
  }
}
}
}
}
}
