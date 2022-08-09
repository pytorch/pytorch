#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
bulirsch_elliptic_integral_el2(T1 x, T1 k_c, T1 a, T1 b) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x) || std::isnan(k_c) || std::isnan(a) || std::isnan(b)) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else {
    return a * x * carlson_elliptic_r_f(T1(1), T1(1) + k_c * k_c * (x * x), T1(1) + x * x)
        + (b - a) * (x * x) * carlson_elliptic_r_d(T1(1), T1(1) + k_c * k_c * (x * x), T1(1) + x * x) / T1(3);
  }
}
}
