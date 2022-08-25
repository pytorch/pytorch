#pragma once

#include <ATen/native/special/detail/numeric_t.h>
#include <c10/util/numbers.h>
#include <ATen/native/special/detail/is_integer.h>
#include <ATen/native/special/sin_pi.h>
#include <ATen/native/special/detail/ln_gamma.h>
#include <ATen/native/special/detail/is_even_integer.h>
#include <ATen/native/special/detail/gamma.h>
#include <ATen/native/special/prime_number.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
riemann_zeta_summation(T1 s) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::real(s) < T3(1)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::real(s) > T3(1)) {
    T2 p = T2(1);
    T2 q;

    for (auto k = 2; k < 10000; k++) {
      q = std::pow(T2(k), -s);
      p = p + q;

      if (std::abs(q) < std::numeric_limits<T3>::epsilon() * std::abs(p) || (std::abs(q) < std::numeric_limits<T3>::epsilon() && std::abs(p) < T3(100) * std::numeric_limits<T3>::epsilon())) {
        break;
      }
    }

    return p;
  } else {
    return std::pow(T3(2) * c10::numbers::pi_v<T3>, s) * sin_pi(T3{0.5L} * s) * gamma(T2(1) - s) * riemann_zeta_summation(T2(1) - s) / c10::numbers::pi_v<T3>;
  }
}
}
}
}
}
