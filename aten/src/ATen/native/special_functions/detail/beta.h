#pragma once

#include <ATen/native/special_functions/detail/gamma.h>
#include <ATen/native/special_functions/detail/ln_gamma.h>
#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
beta(T1 a, T1 b) {
  const auto quiet_nan = std::numeric_limits<T1>::quiet_NaN();

  if (std::isnan(a) || std::isnan(b)) {
    return quiet_nan;
  } else {
    const auto c = a + b;

    const auto abs_a = std::abs(a);
    const auto abs_b = std::abs(b);
    const auto abs_c = std::abs(c);

    const auto size = c10::numbers::factorials_size<T1>;

    const auto nearbyint_a = std::nearbyint(a);
    const auto nearbyint_b = std::nearbyint(b);
    const auto nearbyint_c = std::nearbyint(c);

    const auto int_nearbyint_a = int(nearbyint_a);
    const auto int_nearbyint_b = int(nearbyint_b);
    const auto int_nearbyint_c = int(nearbyint_c);

    const auto positive_a = int_nearbyint_a != a || int_nearbyint_a >= 1;
    const auto positive_b = int_nearbyint_b != b || int_nearbyint_b >= 1;

    const auto negative_c = int_nearbyint_c == c && int_nearbyint_c <= 0;

    if (abs_a < size && abs_b < size && abs_c < size) {
      if (negative_c) {
        if (positive_a || positive_b) {
          return T1(0);
        } else {
          return quiet_nan;
        }
      } else {
        const auto gamma_a = at::native::special_functions::detail::gamma(a);
        const auto gamma_b = at::native::special_functions::detail::gamma(b);
        const auto gamma_c = at::native::special_functions::detail::gamma(c);

        if (abs_b > abs_a) {
          return gamma_b / gamma_c * gamma_a;
        } else {
          return gamma_a / gamma_c * gamma_b;
        }
      }
    } else {
      if (negative_c) {
        if (positive_a || positive_b) {
          return T1(0);
        } else {
          return quiet_nan;
        }
      } else {
        const auto ln_gamma_sign_a = at::native::special_functions::detail::log_gamma_sign(a);
        const auto ln_gamma_sign_b = at::native::special_functions::detail::log_gamma_sign(b);
        const auto ln_gamma_sign_c = at::native::special_functions::detail::log_gamma_sign(c);

        const auto ln_gamma_sign = ln_gamma_sign_a * ln_gamma_sign_b * ln_gamma_sign_c;

        const auto ln_gamma_a = at::native::special_functions::detail::ln_gamma(a);
        const auto ln_gamma_b = at::native::special_functions::detail::ln_gamma(b);
        const auto ln_gamma_c = at::native::special_functions::detail::ln_gamma(c);

        const auto ln_gamma = ln_gamma_a + ln_gamma_b - ln_gamma_c;

        if (ln_gamma > std::log(std::numeric_limits<T1>::max())) {
          return ln_gamma_sign * std::numeric_limits<T1>::infinity();
        } else {
          return ln_gamma_sign * std::exp(ln_gamma);
        }
      }
    }
  }
}
}
