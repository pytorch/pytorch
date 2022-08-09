#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/ln_gamma.h>
#include <ATen/native/special_functions/detail/bessel.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
bessel_j(T1 x, T1 n) {
  using T2 = numeric_t<T1>;

  if (x < T1(0)) {
    throw std::domain_error("negative `x`");
  } else if (std::isnan(n) || std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (n >= T1(0) && x * x < T1(10) * (n + T1(1))) {
    if (std::abs(x) < std::numeric_limits<T2>::epsilon()) {
      if (n == T1(0)) {
        return T1(1);
      } else {
        return T1(0);
      }
    } else {
      auto p = T1(1);
      auto q = T1(1);

      for (unsigned int j = 1; j < 200; j++) {
        q = q * (T1(-1) * (x / T2(2)) * (x / T2(2)) / (T1(j) * (T1(n) + T1(j))));
        p = p + q;

        if (std::abs(q / p) < std::numeric_limits<T2>::epsilon()) {
          break;
        }
      }

      return std::exp(T1(n) * std::log(x / T2(2)) - ln_gamma(T2(1) + n)) * p;
    }
  } else {
    return bessel(x, n).j;
  }
}
}
