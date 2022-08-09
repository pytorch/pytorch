#pragma once

#include <cmath>

#include <ATen/native/special_functions/detail/digamma.h>
#include <ATen/native/special_functions/detail/is_integer.h>
#include <ATen/native/special_functions/detail/ln_gamma.h>
#include <ATen/native/special_functions/detail/hurwitz_zeta.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
polygamma(unsigned int m, T1 x) {
  if (m == 0) {
    return digamma(x);
  } else if (const auto n = is_integer(x); n && n() <= 0) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (m % 2 == 0) {
    return -(std::exp(ln_gamma(T1(m + 1))) * hurwitz_zeta(T1(m + 1), x));
  } else {
    return +(std::exp(ln_gamma(T1(m + 1))) * hurwitz_zeta(T1(m + 1), x));
  }
}
}
