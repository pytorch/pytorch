#pragma once

#include <ATen/native/special_functions/detail/bessel.h>
#include <ATen/native/special_functions/detail/modified_bessel.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
modified_bessel_i(T1 n, T1 x) {
  if (x < T1(0)) {
    throw std::domain_error("modified_bessel_i: Argument < 0");
  } else if (std::isnan(n) || std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (n >= T1(0) && x * x < T1(10) * (n + T1(1))) {
    return detail::regular_bessel_series_expansion(n, x, +1, 200);
  } else {
    return modified_bessel(n, x).i;
  }
}
}
