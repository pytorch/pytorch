#pragma once

#include <ATen/native/special_functions/detail/carlson_elliptic_r_d.h>
#include <ATen/native/special_functions/detail/carlson_elliptic_r_c.h>
#include <ATen/native/special_functions/detail/numeric_t.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
complete_legendre_elliptic_integral_d(T1 k) {
  using T2 = numeric_t<T1>;

  if (std::isnan(k)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    return carlson_elliptic_r_d(T1(0), T1(1) - k * k, T1(1)) / T2(3);
  }
}
}
