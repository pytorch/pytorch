#pragma once

#include <ATen/native/special_functions/detail/ln_gamma.h>
#include <ATen/native/special_functions/detail/ln_gamma_sign.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
incomplete_beta_continued_fraction(T1 a, T1 b, T1 x) {
  auto p = T1(1);
  auto q = T1(1) - (a + b) * x / (a + T1(1));

  const auto minimum = 1000 * std::numeric_limits<T1>::min();

  if (std::abs(q) < minimum) {
    q = minimum;
  }

  q = T1(1) / q;

  auto r = q;

  for (unsigned int j = 1; j <= 100; j++) {
    auto s = T1(j) * (b - T1(j)) * x / ((a - T1(1) + T1(2 * j)) * (a + T1(2 * j)));

    q = T1(1) + s * q;

    if (std::abs(q) < minimum) {
      q = minimum;
    }

    p = T1(1) + s / p;

    if (std::abs(p) < minimum) {
      p = minimum;
    }

    q = T1(1) / q;

    r = r * (q * p);

    s = -(a + T1(j)) * (a + b + T1(j)) * x / ((a + T1(2 * j)) * (a + T1(1) + T1(2 * j)));

    q = T1(1) + s * q;

    if (std::abs(q) < minimum) {
      q = minimum;
    }

    p = T1(1) + s / p;

    if (std::abs(p) < minimum) {
      p = minimum;
    }

    q = T1(1) / q;

    r = r * (q * p);

    if (std::abs(q * p - T1(1)) < std::numeric_limits<T1>::epsilon()) {
      return r;
    }
  }

  throw std::runtime_error("continued fractions error");
}

template<typename T1>
T1
incomplete_beta(T1 a, T1 b, T1 x) {
  if (x < T1(0) || x > T1(1)) {
    throw std::domain_error("incomplete_beta: argument out of range");
  } else {
    if ((std::isnan(x) || std::isnan(a) || std::isnan(b)) || (a == T1(0) && b == T1(0))) {
      return std::numeric_limits<T1>::quiet_NaN();
    } else if (a == T1(0)) {
      if (x > T1(0)) {
        return T1(1);
      } else {
        return T1(0);
      }
    } else if (b == T1(0)) {
      if (x < T1(1)) {
        return T1(0);
      } else {
        return T1(1);
      }
    } else {
      const auto c = a + b;

      const auto ln_gamma_sign_a = at::native::special_functions::detail::log_gamma_sign(a);
      const auto ln_gamma_sign_b = at::native::special_functions::detail::log_gamma_sign(b);
      const auto ln_gamma_sign_c = at::native::special_functions::detail::log_gamma_sign(c);

      const auto ln_gamma_sign = ln_gamma_sign_c * ln_gamma_sign_a * ln_gamma_sign_b;

      const auto ln_gamma_a = at::native::special_functions::detail::ln_gamma(a);
      const auto ln_gamma_b = at::native::special_functions::detail::ln_gamma(b);
      const auto ln_gamma_c = at::native::special_functions::detail::ln_gamma(c);

      const auto ln_gamma = ln_gamma_c - ln_gamma_a - ln_gamma_b;

      const auto log_x = std::log(x);

      if (x < (a + T1(1)) / (c + T1(2))) {
        const auto continued_fraction = incomplete_beta_continued_fraction(a, b, x);

        return ln_gamma_sign * std::exp(ln_gamma + a * log_x + b * std::log(T1(1) - x)) * continued_fraction / a;
      } else {
        const auto continued_fraction = incomplete_beta_continued_fraction(b, a, T1(1) - x);

        return T1(1)
            - ln_gamma_sign * std::exp(ln_gamma + a * log_x + b * std::log(T1(1) - x)) * continued_fraction / b;
      }
    }
  }
}
}
}
}
}
