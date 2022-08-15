#pragma once

#include <ATen/native/special_functions/detail/spherical_bessel.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
std::complex<T1>
spherical_hankel_h_2(unsigned int n, T1 x) {
  using T2 = std::complex<T1>;

  if (std::isnan(x)) {
    return {
        std::numeric_limits<T1>::quiet_NaN(),
        std::numeric_limits<T1>::quiet_NaN(),
    };
  } else if (x < T1(0)) {
    return spherical_bessel_negative_x(n, x).j - T2{0, 1} * spherical_bessel_negative_x(n, x).y;
  } else {
    return {spherical_bessel(n, x).j, -spherical_bessel(n, x).y};
  }
}
}
}
}
}
