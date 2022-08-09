#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/complete_carlson_elliptic_r_f.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
complete_elliptic_integral_k(T1 k) {
  using T2 = numeric_t<T1>;

  if (std::isnan(k) || std::abs(k) == T2(1)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    const auto x = T1(1) - k * k;
    const auto y = T1(1);

    const auto r_f = complete_carlson_elliptic_r_f(x, y);

    return r_f;
  }
}
}
