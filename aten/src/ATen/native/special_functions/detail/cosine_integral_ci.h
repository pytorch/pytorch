#pragma once

#include <cmath>
#include <complex>

#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
cosine_integral_ci(T1 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::abs(x) == T1(0)) {
    return -std::numeric_limits<T1>::infinity();
  } else if (std::abs(x) > T1(1000)) {
    auto p = T1(1);
    auto q = T1(1);

    const auto v = T1(1) / std::abs(x);

    p = p * v;

    auto r = T1(p);
    auto s = T1(1);

    auto t = true;

    unsigned int j = 2;

    while (true) {
      p = p * (j * v);

      if (t) {
        s = -s;
        q = q + (s * p);
      } else {
        r = r + (s * p);

        if (p / std::abs(r) < T1(5) * std::numeric_limits<T1>::epsilon()) {
          break;
        }
      }

      t = !t;

      // if (j > 100) series evaluation error

      j++;
    }

    return std::sin(std::abs(x)) * v * q - std::cos(std::abs(x)) * v * r;
  } else if (std::abs(x) > T1(2)) {
    auto p = std::complex<T1>(T1(1), std::abs(x));
    auto q = std::complex<T1>(T1(1) / std::numeric_limits<T1>::min());
    auto r = std::complex<T1>(T1(1) / p);
    auto s = std::complex<T1>(r);

    int j = 2;

    while (true) {
      p = p + T1(2);
      r = T1(1) / (-T1(j - 1) * T1(j - 1) * r + p);
      q = p + -T1(j - 1) * T1(j - 1) / q;
      s = s * (q * r);

      if (std::abs(q * r - T1(1)) < T1(5) * std::numeric_limits<T1>::epsilon()) {
        break;
      }

      // if (j > 100) continued fraction error

      j++;
    }

    s = s * (std::polar(T1(1), -std::abs(x)));

    return -std::real(s);
  } else if (std::abs(x) * std::abs(x) < std::numeric_limits<T1>::min()) {
    return c10::numbers::egamma_v<T1> + std::log(std::abs(x)) + T1(0);
  } else {
    T1 p(0);
    T1 q(0);
    T1 r(0);
    T1 s(1);
    T1 t(1);
    T1 u(0);
    T1 v(0);

    auto w = true;

    unsigned int k = 1;

    while (true) {
      t = t * (std::abs(x) / k);
      u = t / k;
      r = r + (s * u);
      v = u / std::abs(r);

      if (w) {
        s = -s;
        q = r;
        r = p;
      } else {
        p = r;
        r = q;
      }

      if (v < T1(5) * std::numeric_limits<T1>::epsilon()) {
        break;
      }

      w = !w;

      k++;

      // if (k > 100) series evaluation error
    }

    return c10::numbers::egamma_v<T1> + std::log(std::abs(x)) + p;
  }
}
}
