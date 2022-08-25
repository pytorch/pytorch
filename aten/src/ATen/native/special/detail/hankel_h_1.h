#pragma once

#include <complex>

#include <ATen/native/special/detail/bessel.h>
#include <ATen/native/special/detail/bessel_negative_z.h>
#include <ATen/native/special/polar_pi.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
c10::complex<T1>
hankel_h_1(T1 n, T1 x) {
  using T2 = c10::complex<T1>;

  if (n < T1(0)) {
    return at::native::special::polar_pi(T1(1), -n) * hankel_h_1(-n, x);
  } else if (std::isnan(x)) {
    return T2{std::numeric_limits<T1>::quiet_NaN(), std::numeric_limits<T1>::quiet_NaN()};
  } else if (x < T1(0)) {
    return bessel_negative_z(n, x).j + T2{0, 1} * bessel_negative_z(n, x).y;
  } else {
    return T2{bessel(n, x).j, bessel(n, x).y};
  }
}
}
}
}
}
