#pragma once

#include <ATen/native/special_functions/detail/carlson_elliptic_r_f.h>
#include <ATen/native/special_functions/detail/carlson_elliptic_r_j.h>
#include <ATen/native/special_functions/detail/numeric_t.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
complete_elliptic_integral_pi(T1 n, T1 k) {
  using T2 = numeric_t<T1>;

  if (std::isnan(k) || std::isnan(n)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (n == T1(1)) {
    return std::numeric_limits<T1>::infinity();
  } else if (std::abs(k) > T2(1)) {
    throw std::domain_error("bad `k`");
  } else {
    const auto x = T1(0);
    const auto y = T1(1) - k * k;
    const auto z = T1(1);
    const auto p = T1(1) - n;

    const auto r_f = carlson_elliptic_r_f(x, y, z);
    const auto r_j = carlson_elliptic_r_j(x, y, z, p);

    return r_f + n * r_j / T1(3);
  }
}
}
}
}
}
