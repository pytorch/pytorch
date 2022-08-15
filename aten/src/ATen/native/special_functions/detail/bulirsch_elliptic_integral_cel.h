#pragma once

#include "carlson_elliptic_r_f.h"
#include "carlson_elliptic_r_j.h"

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
bulirsch_elliptic_integral_cel(T1 k_c, T1 p, T1 a, T1 b) {
  if (std::isnan(k_c) || std::isnan(p) || std::isnan(a) || std::isnan(b)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    return a * carlson_elliptic_r_f(T1(0), k_c * k_c, T1(1))
        + (b - p * a) * carlson_elliptic_r_j(T1(0), k_c * k_c, T1(1), p) / T1(3);
  }
}
}
}
}
}
