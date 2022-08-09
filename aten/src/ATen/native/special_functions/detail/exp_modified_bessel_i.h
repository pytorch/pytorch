#pragma once

#include <ATen/native/special_functions/detail/modified_bessel.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
exp_modified_bessel_i(T1 n, T1 x) {
  if (x < T1(0)) {
    throw std::domain_error("`x` < 0");
  } else if (std::isnan(n) || std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    return modified_bessel(n, x, true).i;
  }
}
}
