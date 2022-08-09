#pragma once

#include <cmath>

#include <ATen/native/special_functions/detail/expint.h>
#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
exponential_integral_e(unsigned int n, T1 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (n <= 1 && x == T1(0)) {
    return std::numeric_limits<T1>::infinity();
  } else if (n == 0) {
    return std::exp(-x) / x;
  } else if (n == 1) {
    return expint_E1(x);
  } else if (x == T1(0)) {
    return T1(1) / static_cast<T1>(n - 1);
  } else if (x < T1(1)) {
    T1 p = (n - 1 != 0 ? T1(1) / (n - 1) : -std::log(x) - c10::numbers::egamma_v<T1>);
    T1 q = T1(1);
    T1 r;

    for (unsigned int j = 1; j <= 1000; j++) {
      q = q * (-x / T1(j));

      if (int(j) != n - 1) {
        r = -q / T1(j - (n - 1));
      } else {
        T1 s = -c10::numbers::egamma_v<T1>;

        for (int k = 1; k <= n - 1; ++k) {
          s = s + (T1(1) / T1(k));
        }

        r = q * (s - std::log(x));
      }

      p = q + r;

      if (std::abs(r) < std::numeric_limits<T1>::epsilon() * std::abs(p)) {
        return p;
      }
    }

    throw std::runtime_error("series summation error");
  } else if (n > 50000) {
    auto p = T1(1);
    auto q = T1(1);

    for (unsigned int j = 1; j <= n; j++) {
      p = p * ((n - 2 * (j - 1) * x) / ((x + n) * (x + n)));

      if (std::abs(p) < std::numeric_limits<T1>::epsilon() * std::abs(q)) {
        break;
      }

      q = q + p;
    }

    return std::exp(-x) * q / (x + n);
  } else if (x > T1(100)) {
    auto p = T1(1);
    auto q = T1(1);

    for (unsigned int j = 1; j <= n; j++) {
      const auto previous_p = p;

      p = p * (-T1(n - j + 1) / x);

      if (std::abs(p) > std::abs(previous_p)) {
        break;
      }

      q = q + p;
    }

    return std::exp(-x) * q / x;
  } else {
    auto q = x + T1(n);
    auto r = T1(1) / (T1(4) * std::numeric_limits<T1>::min());
    auto s = T1(1) / q;
    auto t = s;

    for (unsigned int j = 1; j <= 1000; j++) {
      auto p = -T1(j * (n - 1 + j));

      q = q + T1(2);
      s = T1(1) / (p * s + q);

      if (std::abs(s) < T1(4) * std::numeric_limits<T1>::min()) {
        s = std::copysign(T1(4) * std::numeric_limits<T1>::min(), s);
      }

      r = q + p / r;

      if (std::abs(r) < T1(4) * std::numeric_limits<T1>::min()) {
        r = std::copysign(T1(4) * std::numeric_limits<T1>::min(), r);
      }

      t = t * (r * s);

      if (std::abs(r * s - T1(1)) < std::numeric_limits<T1>::epsilon()) {
        return t * std::exp(-x);
      }
    }

    throw std::runtime_error("continued fraction error");
  }
}
}
