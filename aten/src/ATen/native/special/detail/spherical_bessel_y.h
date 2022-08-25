#pragma once

#include <ATen/native/special/detail/spherical_bessel.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
inline constexpr
T1
spherical_bessel_y(unsigned int n, T1 x) {
  if (x < T1(0) || (std::isnan(x))) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (x == T1(0)) {
    return -std::numeric_limits<T1>::infinity();
  } else {
    return spherical_bessel(n, x).y;
  }
}
}
}
}
}
