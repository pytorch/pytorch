#pragma once

#include <ATen/native/special_functions/airy_ai.h>
#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename Tp>
struct hermite_t {
  unsigned int n;
  Tp x;
  Tp H_n;
  Tp H_nm1;
  Tp H_nm2;

  constexpr Tp
  deriv() const noexcept { return Tp(2 * n) * H_nm1; }

  constexpr Tp
  deriv2() const noexcept { return Tp(4 * n * (n - 1)) * H_nm2; }
};

template<typename T1>
T1
hermite_asymptotic_expansion(unsigned int n, T1 x) {
  const auto s_pi = c10::numbers::pi_v<T1>;
  const auto s_sqrt_2 = c10::numbers::sqrt2_v<T1>;
  const auto s_sqrt_2pi = c10::numbers::sqrttau_v<T1>;
  const auto xturn = std::sqrt(T1(2 * n));
  if (std::abs(x - xturn) < T1{0.05L} * xturn) {
    const auto n_2 = T1(n) / T1(2);
    const auto n6th = std::pow(T1(n), T1(1) / T1(6));
    const auto exparg = n * std::log(xturn) - T1(3) * n_2 + xturn * x;
    const auto airyarg = s_sqrt_2 * (x - xturn) * n6th;
    auto Ai = at::native::special_functions::airy_ai(airyarg);
    return s_sqrt_2pi * n6th * std::exp(exparg) * Ai;
  } else if (x < xturn) {
    const auto theta = std::asin(x / xturn);
    const auto _2theta = T1(2) * theta;
    const auto n_2 = T1(n) / T1(2);
    const auto exparg = n_2 * (T1(2) * std::log(xturn) - std::cos(_2theta));
    const auto arg = theta / T1(2) + n_2 * (std::sin(_2theta) + _2theta - s_pi);
    return std::sqrt(T1(2) / std::cos(theta)) * std::exp(exparg) * std::cos(arg);
  } else {
    const auto sigma = std::sqrt((x - xturn) * (x + xturn));
    const auto exparg = T1{0.5L} * (x * (x - sigma) - n) + n * std::log(sigma + x);
    return std::exp(exparg) * std::sqrt(T1{0.5L} * (T1(1) + x / sigma));
  }
}

template<typename T1>
hermite_t<T1>
hermite_recurrence(unsigned int n, T1 x) {
  if (n == 0) {
    return {n, x, T1(1), T1(0), T1(0)};
  } else if (n == 1) {
    return {n, x, T1(2) * x, T1(1), T1(0)};
  } else {
    {
      constexpr auto s_inf = std::numeric_limits<T1>::infinity();
      constexpr auto s_max = std::numeric_limits<T1>::max();

      if (std::abs(x) > std::pow(T1(2), std::log2(s_max) / n - T1{0.125})) {
        return {n, x, T1(n & 1 ? -1 : +1) * s_inf, -(T1(n & 1 ? -1 : +1) * s_inf), T1(n & 1 ? -1 : +1) * s_inf};
      }
    }

    auto p = T1(2) * x;
    auto q = T1(1);
    auto r = T1(2) * (x * p - q);

    for (unsigned int j = 3; j <= n; j++) {
      q = p;
      p = r;
      r = T1(2) * (x * p - T1(j - 1) * q);
    }

    return {n, x, r, p, q};
  }
}

template<typename T1>
T1
hermite_polynomial_h(unsigned int n, T1 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (x < T1(0)) {
    if (n % 2 == 1) {
      return -1 * hermite_polynomial_h(n, -x);
    } else {
      return +1 * hermite_polynomial_h(n, -x);
    }
  } else if (n > 10000) {
    return hermite_asymptotic_expansion(n, x);
  } else {
    return hermite_recurrence(n, x).H_n;
  }
}
}
