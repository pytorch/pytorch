#pragma once

#include <ATen/native/special_functions/detail/bernoulli_number.h>
#include <ATen/native/special_functions/detail/harmonic_number.h>
#include <ATen/native/special_functions/detail/is_half_odd_integer.h>
#include <ATen/native/special_functions/detail/is_integer.h>
#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/tan_pi.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
digamma(unsigned int n) {
  using T2 = numeric_t<T1>;

  if (n > 1) {
    return -c10::numbers::egamma_v<T2> + harmonic_number<T2>(n - 1);
  } else {
    return -c10::numbers::egamma_v<T2>;
  }
}

template<typename T1>
T1
digamma(T1 x) {
  using T2 = numeric_t<T1>;

  const auto is_integer_x = is_integer(x);

  if (std::real(x) <= T2(0)) {
    if (is_integer_x) {
      return std::numeric_limits<T1>::quiet_NaN();
    } else {
      return digamma(T2(1) - x) - c10::numbers::pi_v<T2> / at::native::special_functions::tan_pi(x);
    }
  } else if (is_integer_x) {
    return digamma<T1>(is_integer_x());
  } else if (is_half_odd_integer(x)) {
    T1 p = -c10::numbers::egamma_v<T2> - T1(2) * c10::numbers::ln2_v<T2>;

    for (int j = 1; j <= is_half_odd_integer(x)(); j++) {
      p = p + (T1(2) / T1(2 * j - 1));
    }

    return p;
  } else if (std::real(x) > T2(20)) {
    auto p = std::log(x) - T2(0.5L) / x;
    auto q = x * x;

    for (unsigned int k = 1; k < 100; k++) {
      const auto r = bernoulli_number<T2>(2 * k);
      const auto s = T2(2 * k) * q;

      p = p - (r / s);

      if (std::abs(r / s / p) < std::numeric_limits<T2>::epsilon()) {
        break;
      }

      q = q * (x * x);
    }

    return p;
  } else {
    auto p = T1(0);
    auto q = x;

    while (std::real(q) <= T2(20)) {
      p = p + T2(1) / q;
      q = q + T2(1);
    }

    auto r = std::log(q) - T2(0.5L) / q;
    auto s = q * q;

    for (unsigned int k = 1; k < 100; k++) {
      const auto t = 2 * k;
      const auto u = bernoulli_number<T2>(t);
      const auto v = T2(t) * s;

      r = r - (u / v);

      if (std::abs(u / v / r) < std::numeric_limits<T2>::epsilon()) {
        break;
      }

      s = s * (q * q);
    }

    return r - p;
  }
}
}
}
}
}
