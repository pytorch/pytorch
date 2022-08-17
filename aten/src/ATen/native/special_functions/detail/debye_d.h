#pragma once

#include <ATen/native/special_functions/detail/factorial.h>
#include <ATen/native/special_functions/detail/riemann_zeta.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
debye_d(unsigned int n, T1 x) {
  if (std::isnan(x)) { return std::numeric_limits<T1>::quiet_NaN(); }
  else if (n < 1) { throw std::domain_error("debye_d: Degree n must be positive."); }
  else if (x >= T1(3)) {
    auto sum = T1(0);
    if (n < c10::numbers::factorials_size<T1>())
      sum += factorial<T1>(n) * riemann_zeta<T1>(n + 1);
    else
      return std::numeric_limits<T1>::infinity();

    auto term = T1(0);
    auto expmkx = T1(1);

    for (unsigned int k = 1; k < 100; k++) {
      const auto kx = k * x;
      expmkx *= std::exp(-x);
      auto ksum = T1(1);
      auto kterm = T1(n) * ksum / kx;
      for (unsigned int m = 1; m <= n; ++m) { ksum += std::exchange(kterm, T1(n - m) * kterm / kx); }
      term -= expmkx * ksum * std::pow(x, T1(n)) / T1(k);
    }
    sum += term;
    return T1(n) * sum / std::pow(x, T1(n));
  } else if (std::abs(x) < T1(2) * c10::numbers::pi_v<T1>) {
    auto x2pi2k = x * c10::numbers::inv_tau_v<T1> * (x * c10::numbers::inv_tau_v<T1>);
    auto sum = T1(0);

    for (unsigned int k = 1; k < 200; k++) {
      sum += T1(2) * riemann_zeta<T1>(2 * k) * x2pi2k / T1(2 * k + n);
      if (std::abs(T1(2) * riemann_zeta<T1>(2 * k) * x2pi2k / T1(2 * k + n))
          < std::numeric_limits<T1>::epsilon() * std::abs(sum)) { break; }
      x2pi2k *= -(x * c10::numbers::inv_tau_v<T1> * (x * c10::numbers::inv_tau_v<T1>));
    }

    sum *= T1(n);
    sum += T1(1) - T1(n) * x / T1(2 * (n + 1));
    return sum;
  } else {
    return T1(0);
  }
}
}
}
}
}
