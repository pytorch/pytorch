#pragma once

#include <cmath>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename Tp>
struct gegenbauer_polynomial_t {
  unsigned int n;
  Tp lambda;
  Tp x;
  Tp C_n;
  Tp C_nm1;
  Tp C_nm2;

  constexpr Tp
  deriv() const noexcept {
    auto apbp2k = Tp{2} * lambda + Tp(2 * n);
    return (n * (-apbp2k * x) * C_nm1
        + Tp{2} * (n + lambda) * (n + lambda) * C_nm2)
        / (apbp2k * (Tp{1} - x * x));
  }
};

template<typename T1>
gegenbauer_polynomial_t<T1>
gegenbauer_polynomial_c(unsigned int n, T1 lambda, T1 x) {
  if (std::isnan(lambda) || std::isnan(x)) {
    return {n, lambda, x, std::numeric_limits<T1>::quiet_NaN(), std::numeric_limits<T1>::quiet_NaN(),
            std::numeric_limits<T1>::quiet_NaN()};
  } else if (n == 0) {
    return {n, lambda, x, T1(1), T1(0), T1(0)};
  } else if (n == 1) {
    return {n, lambda, x, T1(2) * lambda * x, T1(1), T1(0)};
  } else {
    auto p = T1(2) * lambda * x;
    auto q = T1(1);
    auto r = (T1(2) * (T1(1) + lambda) * x * p - T1(2) * lambda * q) / T1(2);

    for (unsigned int j = 3; j <= n; j++) {
      q = p;
      p = r;
      r = (T1(2) * (T1(j) - T1(1) + lambda) * x * p - (T1(j) - T1(2) + T1(2) * lambda) * q) / T1(j);
    }

    return {n, lambda, x, r, p, q};
  }
}
}
}
}
}
