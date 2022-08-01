#pragma once

#include <ATen/native/math/numeric_t.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
inline constexpr
T1
elliptic_theta_3(T1 n, T1 x) {
  using T2 = numeric_t<T1>;

  const auto epsilon = std::numeric_limits<T1>::epsilon();

  const auto pi_v = c10::pi<T2>;

  if (std::isnan(n) || std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::abs(x) <= T2(1) / pi_v) {
    auto p = std::exp(-n * n / x);

    for (auto k = 1; k < 20; k++) {
      const auto positive = n + T1(k);
      const auto negative = n - T1(k);

      const auto exp_positive = std::exp(-positive * positive / x);
      const auto exp_negative = std::exp(-negative * negative / x);

      const auto abs_p = std::abs(p);

      const auto abs_exp_positive = std::abs(exp_positive);
      const auto abs_exp_negative = std::abs(exp_negative);

      p = p + (exp_positive + exp_negative);

      if (abs_exp_positive < epsilon * abs_p && abs_exp_negative < epsilon * abs_p) {
        break;
      }
    }

    return p / std::sqrt(pi_v * x);
  } else {
    auto p = T1(0);

    for (auto k = 1; k < 20; k++) {
      const auto positive = T1(2 * k);

      const auto positive_pi_v = positive * pi_v;

      const auto exp_positive = std::exp(-1 * (positive_pi_v * positive_pi_v * x / T1(4)));
      const auto cos_positive = std::cos(+1 * (n * positive_pi_v));

      const auto exp_cos_positive = exp_positive * cos_positive;

      p = p + exp_cos_positive;

      if (std::abs(exp_cos_positive) < epsilon * std::abs(p)) {
        break;
      }
    }

    return T1(1) + T1(2) * p;
  }
} // T1 elliptic_theta_3(T1 n, T1 x)
} // namespace detail
} // namespace special_functions
} // namespace native
} // namespace at
