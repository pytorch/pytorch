#pragma once

#include "carlson_elliptic_r_f.h"
#include "carlson_elliptic_r_j.h"

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
bulirsch_elliptic_integral_el3(T1 x, T1 k_c, T1 p) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x) || std::isnan(k_c) || std::isnan(p)) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else {
    return x * carlson_elliptic_r_f(T2(1), T1(1) + k_c * k_c * (x * x), T1(1) + x * x) + (T1(1) - p) * (x * x)
        * carlson_elliptic_r_j(T1(1), T1(1) + k_c * k_c * (x * x), T1(1) + x * x, T1(1) + p * (x * x)) / T1(3);
  }
}
}
}
}
}
