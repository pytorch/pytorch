#pragma once

#include <ATen/native/special/detail/gamma.h>
#include <ATen/native/special/detail/is_even_integer.h>
#include <ATen/native/special/detail/is_integer.h>
#include <ATen/native/special/detail/ln_gamma.h>
#include <ATen/native/special/detail/numeric_t.h>
#include <ATen/native/special/detail/riemann_zeta_convergent_series.h>
#include <ATen/native/special/detail/riemann_zeta_m_1.h>
#include <ATen/native/special/detail/riemann_zeta_summation.h>
#include <ATen/native/special/prime_number.h>
#include <ATen/native/special/sin_pi.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
riemann_zeta(T1 s) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::isnan(s)) {
    return std::numeric_limits<T3>::quiet_NaN();
  } else if (s == T2(1)) {
    return std::numeric_limits<T3>::infinity();
  } else if (is_integer(s) && is_integer(s)() >= 0) {
    return T3(1) + riemann_zeta_m_1(T3(is_integer(s)()));
  } else if (is_integer(s) && is_integer(s)() < 0) {
    return std::pow(T3(2) * c10::numbers::pi_v<T3>, T3(is_integer(s)())) * sin_pi(T3{0.5L} * T3(is_integer(s)())) * gamma(T3(1) - T3(is_integer(s)())) * (T3(1) + riemann_zeta_m_1(T3(1) - T3(is_integer(s)()))) / c10::numbers::pi_v<T3>;
  } else if (std::real(s) < -T3(19)) {
    auto p = T2(1);

    for (auto j = 0; j < 10000; j++) {
      p = p * (T2(1) - std::pow(T3(at::native::special::prime_number<T2>(j)), -(T2(1) - s)));

      if (std::abs(T1(1) - (T2(1) - std::pow(T3(at::native::special::prime_number<T2>(j)), -(T2(1) - s)))) < std::numeric_limits<T3>::epsilon()) {
        break;
      }
    }

    p = T1(1) / p;

    return p * (p * (std::pow(T3(2) * c10::numbers::pi_v<T3>, s) * sin_pi(T3(0.5L) * s) * std::exp(ln_gamma(T2(1) - s)) / c10::numbers::pi_v<T3>));
  } else if (std::real(s) < std::numeric_limits<T3>::digits) {
    return riemann_zeta_convergent_series(s);
  } else {
    constexpr auto is_not_complex = is_complex_v<T2>;

    if (is_not_complex) {
      return T2(1) + std::pow(numeric_t<T2>(2), -s);
    } else {
      return T2(1) + std::exp2(-s);
    }
  }
}
}
}
}
}
