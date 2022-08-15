#pragma once

#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
sinh_pi(T1 x) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (x < T1(0)) {
    return -sinh_pi(-x);
  } else {
    return std::sinh(c10::numbers::pi_v<T2> * x);
  }
}

template<typename T1>
std::complex<T1>
sinh_pi(std::complex<T1> z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  return std::sinh(c10::numbers::pi_v<T3> * std::real(z)) * cos_pi(std::imag(z))
      + std::complex<T1>{0, 1} * std::cosh(c10::numbers::pi_v<T3> * std::real(z)) * sin_pi(std::imag(z));
}
}
}
}
}
