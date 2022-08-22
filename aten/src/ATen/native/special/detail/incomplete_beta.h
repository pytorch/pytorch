#pragma once

#include <ATen/native/special/detail/ln_gamma.h>
#include <ATen/native/special/detail/ln_gamma_sign.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
incomplete_beta_continued_fraction(T1 a, T1 b, T1 x) {
  auto p = T1(1);
  auto q = T1(1) - (a + b) * x / (a + T1(1));

  if (std::abs(q) < 1000 * std::numeric_limits<T1>::min()) {
    q = 1000 * std::numeric_limits<T1>::min();
  }

  q = T1(1) / q;

  auto r = q;

  for (unsigned int j = 1; j <= 100; j++) {
    auto s = T1(j) * (b - T1(j)) * x / ((a - T1(1) + T1(2 * j)) * (a + T1(2 * j)));

    q = T1(1) + s * q;

    if (std::abs(q) < 1000 * std::numeric_limits<T1>::min()) {
      q = 1000 * std::numeric_limits<T1>::min();
    }

    p = T1(1) + s / p;

    if (std::abs(p) < 1000 * std::numeric_limits<T1>::min()) {
      p = 1000 * std::numeric_limits<T1>::min();
    }

    q = T1(1) / q;

    r = r * (q * p);

    s = -(a + T1(j)) * (a + b + T1(j)) * x / ((a + T1(2 * j)) * (a + T1(1) + T1(2 * j)));

    q = T1(1) + s * q;

    if (std::abs(q) < 1000 * std::numeric_limits<T1>::min()) {
      q = 1000 * std::numeric_limits<T1>::min();
    }

    p = T1(1) + s / p;

    if (std::abs(p) < 1000 * std::numeric_limits<T1>::min()) {
      p = 1000 * std::numeric_limits<T1>::min();
    }

    q = T1(1) / q;

    r = r * (q * p);

    if (std::abs(q * p - T1(1)) < std::numeric_limits<T1>::epsilon()) {
      return r;
    }
  }

  return std::numeric_limits<T1>::quiet_NaN();
}

template<typename T1>
T1
incomplete_beta(T1 a, T1 b, T1 x) {
  if (x < T1(0) || x > T1(1)) {
    return std::numeric_limits<T1>::quiet_NaN();
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
      if (x < (a + T1(1)) / (a + b + T1(2))) {
        return log_gamma_sign(a + b) * log_gamma_sign(a) * log_gamma_sign(b) * std::exp(ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * std::log(x) + b * std::log(T1(1) - x)) * incomplete_beta_continued_fraction(a, b, x) / a;
      } else {
        return T1(1) - log_gamma_sign(a + b) * log_gamma_sign(a) * log_gamma_sign(b) * std::exp(ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * std::log(x) + b * std::log(T1(1) - x)) * incomplete_beta_continued_fraction(b, a, T1(1) - x) / b;
      }
    }
  }
}
}
}
}
}
