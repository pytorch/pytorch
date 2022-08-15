#pragma once

#include <complex>

#include <ATen/native/special_functions/detail/spherical_bessel.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
std::complex<T1>
spherical_hankel_h_1(unsigned int n, T1 x) {
  using T2 = std::complex<T1>;

  if (std::isnan(x)) {
    return T2{std::numeric_limits<T1>::quiet_NaN(), std::numeric_limits<T1>::quiet_NaN()};
  } else if (x < T1(0)) {
    const auto a = spherical_bessel_negative_x(n, x);

    return a.j + T2{0, 1} * a.y;
  } else {
    return {spherical_bessel(n, x).j, spherical_bessel(n, x).y};
  }
}
}
}
}
}
