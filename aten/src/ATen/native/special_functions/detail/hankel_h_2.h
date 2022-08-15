#pragma once

#include <complex>

#include <ATen/native/special_functions/detail/bessel.h>
#include <ATen/native/special_functions/polar_pi.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
std::complex<T1>
hankel_h_2(T1 n, T1 x) {
  using T2 = std::complex<T1>;

  if (n < T1(0)) {
    return at::native::special_functions::polar_pi(T1(1), n) * hankel_h_2(-n, x);
  } else if (std::isnan(x)) {
    return T2{std::numeric_limits<T1>::quiet_NaN(), std::numeric_limits<T1>::quiet_NaN()};
  } else if (x < T1(0)) {
    return bessel_negative_x(n, x).j - T2{0, 1} * bessel_negative_x(n, x).y;
  } else {
    return {bessel(x, n).j, -bessel(x, n).y};
  }
}
}
}
}
}
