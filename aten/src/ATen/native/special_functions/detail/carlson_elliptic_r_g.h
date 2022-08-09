#pragma once

#include "carlson_elliptic_r_d.h"
#include "complete_carlson_elliptic_r_g.h"

namespace at::native::special_functions::detail {
template<typename T1>
T1
carlson_elliptic_r_g(T1 x, T1 y, T1 z) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x) || std::isnan(y) || std::isnan(z)) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else if (z == T1(0)) {
    return complete_carlson_elliptic_r_g(x, y);
  } else if (x == T1(0)) {
    return complete_carlson_elliptic_r_g(y, z);
  } else if (y == T1(0)) {
    return complete_carlson_elliptic_r_g(z, x);
  } else {
    const auto p = x * (y + z) * carlson_elliptic_r_d(y, z, x);
    const auto q = y * (z + x) * carlson_elliptic_r_d(z, x, y);
    const auto r = z * (x + y) * carlson_elliptic_r_d(x, y, z);

    return (p + q + r) / T2(6);
  }
}
}
