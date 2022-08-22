#pragma once

#include <ATen/native/special/detail/bernoulli_number.h>
#include <ATen/native/special/detail/harmonic_number.h>
#include <ATen/native/special/detail/is_half_odd_integer.h>
#include <ATen/native/special/detail/is_integer.h>
#include <ATen/native/special/detail/numeric_t.h>
#include <ATen/native/special/detail/promote_t.h>
#include <ATen/native/special/tan_pi.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
digamma(unsigned int z) {
  using T2 = numeric_t<T1>;

  if (z > 1) {
    return -c10::numbers::egamma_v<T2> + harmonic_number<T2>(z - 1);
  } else {
    return -c10::numbers::egamma_v<T2>;
  }
}

template<typename T1>
T1
digamma(T1 z) {
  using T2 = numeric_t<T1>;

  if (std::real(z) <= T2(0)) {
    if (is_integer(z)) {
      return std::numeric_limits<T1>::quiet_NaN();
    } else {
      return digamma(T2(1) - z) - c10::numbers::pi_v<T2> / at::native::special::tan_pi(z);
    }
  } else if (is_integer(z)) {
    return digamma<T1>(is_integer(z)());
  } else if (is_half_odd_integer(z)) {
    auto p = -c10::numbers::egamma_v<T2> - T1(2) * c10::numbers::ln2_v<T2>;

    for (int j = 1; j <= is_half_odd_integer(z)(); j++) {
      p = p + (T1(2) / T1(2 * j - 1));
    }

    return p;
  } else if (std::real(z) > T2(20)) {
    auto p = std::log(z) - T2(0.5L) / z;
    auto q = z * z;

    for (unsigned int k = 1; k < 100; k++) {
      const auto r = bernoulli_number<T2>(2 * k);
      const auto s = T2(2 * k) * q;

      p = p - (r / s);

      if (std::abs(r / s / p) < std::numeric_limits<T2>::epsilon()) {
        break;
      }

      q = q * (z * z);
    }

    return p;
  } else {
    auto p = T1(0);
    auto q = z;

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
