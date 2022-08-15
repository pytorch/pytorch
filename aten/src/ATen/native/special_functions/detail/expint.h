#pragma once

#include <cmath>

#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename Tp>
Tp expint_E1(Tp);

template<typename T1>
T1
expint_Ei(T1 x) {
  if (x < T1(0)) {
    return -expint_E1(-x);
  } else if (x < -std::log(std::numeric_limits<T1>::epsilon())) {
    T1 p = T1(1);
    T1 q = T1(0);

    for (unsigned int j = 1; j < 1000; j++) {
      p = p * (x / j);
      q = q + (p / j);

      if (std::abs(p) < std::numeric_limits<T1>::epsilon() * std::abs(q)) {
        break;
      }
    }

    return c10::numbers::egamma_v<T1> + q + std::log(x);
  } else {
    T1 p = T1(1);
    T1 q = T1(1);

    for (unsigned int j = 1; j < 1000; j++) {
      const auto previous_p = p;

      p = p * (j / x);

      if (std::abs(p) >= std::abs(previous_p)) {
        break;
      }

      q = q + p;

      if (std::abs(p) < std::numeric_limits<T1>::epsilon() * std::abs(q)) {
        break;
      }
    }

    return std::exp(x) * q / x;
  }
}

template<typename T1>
T1
expint_E1(T1 x) {
  if (x < T1(0)) {
    return -expint_Ei(-x);
  } else if (x < T1(1)) {
    auto p = T1(1);
    auto q = T1(0);
    auto r = T1(0);

    for (unsigned int j = 1; j < 1000; j++) {
      p = p * (-x / j);

      if (std::abs(p) < std::numeric_limits<T1>::epsilon() * std::min(std::abs(q), std::abs(r))) {
        break;
      }

      if (p >= T1(0)) {
        q = q + (p / j);
      } else {
        r = r + (p / j);
      }
    }

    return -q - r - c10::numbers::egamma_v<T1> - std::log(x);
  } else if (x < T1(100)) {
    auto n = 1;

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
  } else {
    auto p = T1(1);
    auto q = T1(1);
    auto r = T1(0);

    for (unsigned int j = 1; j < 1000; j++) {
      const auto previous_p = p;

      p = p * (-j / x);

      if (std::abs(p) > std::abs(previous_p)) {
        break;
      }

      if (p >= T1(0)) {
        q = q + p;
      } else {
        r = r + p;
      }
    }

    return std::exp(-x) * (q + r) / x;
  }
}
}
}
}
}
