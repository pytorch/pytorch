#pragma once

#include <ATen/native/special/detail/numeric_t.h>
#include <c10/util/numbers.h>
#include <ATen/native/special/detail/is_integer.h>
#include <ATen/native/special/sin_pi.h>
#include <ATen/native/special/detail/ln_gamma.h>
#include <ATen/native/special/detail/is_even_integer.h>
#include <ATen/native/special/detail/riemann_zeta_m_1.h>
#include <ATen/native/special/detail/riemann_zeta_summation.h>
#include <ATen/native/special/detail/gamma.h>
#include <ATen/native/special/prime_number.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
riemann_zeta_convergent_series(T1 s) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::real(s) < T3(0)) {
    if (is_even_integer(s)) {
      return T2(0);
    } else {
      return riemann_zeta_convergent_series(T2(1) - s) * (std::pow(T3(2) * c10::numbers::pi_v<T3>, s) * sin_pi(T3{0.5L} * s) * gamma(T2(1) - s) / c10::numbers::pi_v<T3>);
    }
  } else {
    auto a = T2(0);
    auto b = T3(0.25L);

    for (auto j = 1; j < 10000; j++) {
      bool c = false;
      auto d = T3(1);
      auto e = T2(0);

      for (auto k = 1; k <= j; k++) {
        d = d * (-T3(j - k + 1) / T3(k));

        if (std::abs(d) > std::exp(std::numeric_limits<T3>::max_exponent10 * std::log(T3(10)) - T3(1))) {
          c = true;

          break;
        }

        e = e + (d * std::pow(T2(1 + k), -s));
      }

      if (c) {
        break;
      }

      e = e * b;
      a = a + e;

      if (std::abs(e) < std::numeric_limits<T3>::epsilon() * std::abs(a) || (std::abs(e) < std::numeric_limits<T3>::epsilon() && std::abs(a) < T3(100) * std::numeric_limits<T3>::epsilon())) {
        break;
      }

      b = b * T3(0.5L);
    }

    return T1(1) + ((a + (std::pow(T2(2), T2(1) - s))) / (T2(1) - std::pow(T2(2), T2(1) - s)));
  }
}
}
}
}
}
