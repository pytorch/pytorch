#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
bulirsch_elliptic_integral_el1(T1 x, T1 k_c) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x) || std::isnan(k_c)) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else {
    return x * carlson_elliptic_r_f(T1(1), T1(1) + k_c * k_c * (x * x), T1(1) + x * x);
  }
}
}
}
}
}
