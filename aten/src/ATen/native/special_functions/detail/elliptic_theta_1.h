#pragma once

#include <ATen/native/special_functions/detail/is_zero.h>
#include <ATen/native/special_functions/detail/numeric_t.h>
#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename T1>
inline constexpr T1
elliptic_theta_1(T1 n, T1 x) {
  using T2 = numeric_t<T1>;

  n = n - T1(0.5L);

  if (std::isnan(n) || std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (is_zero(x)) {
    return T1(0);
  } else if (std::abs(x) <= T2(1) / c10::numbers::pi_v<T2>) {
    auto p = std::exp(-n * n / x);
    auto q = T1(-1);

    for (auto j = 1; j < 20; j++) {
      p = +1 * (p + (q * std::exp(-(n + T1(j)) * (n + T1(j)) / x) + q * std::exp(-(n - T1(j)) * (n - T1(j)) / x)));
      q = -1 * q;

      if (std::abs(q * std::exp(-(n + T1(j)) * (n + T1(j)) / x)) < std::numeric_limits<T1>::epsilon() * std::abs(p)
          && std::abs(q * std::exp(-(n - T1(j)) * (n - T1(j)) / x))
              < std::numeric_limits<T1>::epsilon() * std::abs(p)) {
        break;
      }
    }

    return p / std::sqrt(c10::numbers::pi_v<T2> * x);
  } else {
    auto p = T1(0);

    for (auto j = 0; j < 20; j++) {
      p = p + std::exp(
          -1 * (T1(2 * j + 1) * c10::numbers::pi_v<T2> * (T1(2 * j + 1) * c10::numbers::pi_v<T2>) * x / T1(4)))
          * std::cos(+1 * (n * (T1(2 * j + 1) * c10::numbers::pi_v<T2>)));

      if (std::abs(
          std::exp(-1 * (T1(2 * j + 1) * c10::numbers::pi_v<T2> * (T1(2 * j + 1) * c10::numbers::pi_v<T2>) * x / T1(4)))
              * std::cos(+1 * (n * (T1(2 * j + 1) * c10::numbers::pi_v<T2>))))
          < std::numeric_limits<T1>::epsilon() * std::abs(p)) {
        break;
      }
    }

    return T1(2) * p;
  }
}
}
