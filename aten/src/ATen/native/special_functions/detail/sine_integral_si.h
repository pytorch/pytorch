#pragma once

#include <complex>

#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
sine_integral_si(T1 x) {
  const auto abs_x = std::abs(x);

  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (abs_x == T1(0)) {
    if (x < T1(0)) {
      return -T1(0);
    } else {
      return +T1(0);
    }
  } else if (abs_x > T1(1000)) {
    auto p = T1(1);
    auto q = T1(1);

    unsigned int v = T1(1) / abs_x;

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

    if (x < T1(0)) {
      return -(c10::numbers::pi_v<T1> / T1(2) - std::cos(abs_x) * v * q - std::sin(abs_x) * v * r);
    } else {
      return +(c10::numbers::pi_v<T1> / T1(2) - std::cos(abs_x) * v * q - std::sin(abs_x) * v * r);
    }
  } else if (abs_x > T1(2)) {
    std::complex<T1> b(T1(1), abs_x);
    std::complex<T1> c(T1(1) / std::numeric_limits<T1>::min());
    std::complex<T1> d(T1(1) / b);
    std::complex<T1> h(d);

    int j = 2;

    while (true) {

      b = b + T1(2);
      d = T1(1) / (-T1(j - 1) * T1(j - 1) * d + b);
      c = b + -T1(j - 1) * T1(j - 1) / c;
      h = h * (c * d);

      if (std::abs(c * d - T1(1)) < T1(5) * std::numeric_limits<T1>::epsilon()) { break; }

      // if (j > 100) continued fraction error

      j++;
    }

    h = h * (std::polar(T1(1), -abs_x));

    if (x < T1(0)) {
      return -(c10::numbers::pi_v<T1> / T1(2) + std::imag(h));
    } else {
      return +(c10::numbers::pi_v<T1> / T1(2) + std::imag(h));
    }
  } else {
    T1 p(0);
    T1 q(0);
    T1 r(0);
    T1 s(1);
    T1 t(1);

    bool odd = true;

    unsigned int j = 1;

    if (abs_x * abs_x < std::numeric_limits<T1>::min()) {
      q = abs_x;
    } else {
      while (true) {
        t *= abs_x / j;
        T1 u = t / j;
        r += s * u;
        T1 v = u / std::abs(r);

        if (odd) {
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

        odd = !odd;

        j++;

        // if (j > 100) series evaluation error
      }
    }

    if (x < T1(0)) {
      return -q;
    } else {
      return +q;
    }
  }
}
}
}
}
}
