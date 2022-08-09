#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/carlson_elliptic_r_f.h>
#include <ATen/native/special_functions/detail/carlson_elliptic_r_d.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
complete_elliptic_integral_e(T1 k) {
  using T2 = numeric_t<T1>;

  const auto abs_k = std::abs(k);

  if (std::isnan(k)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (abs_k == T2(1)) {
    return T1(1);
  } else if (abs_k > T2(1)) {
    throw std::domain_error("bad `k`");
  } else {
    const auto x = T1(0);
    const auto y = T1(1) - k * k;
    const auto z = T1(1);

    const auto r_f = carlson_elliptic_r_f(x, y, z);
    const auto r_d = carlson_elliptic_r_d(x, y, z);

    return r_f - k * k * r_d / T1(3);
  }
}
}
