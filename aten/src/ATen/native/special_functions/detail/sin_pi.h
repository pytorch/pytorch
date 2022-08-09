#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
sin_pi(T1 x) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (x < T1(0)) {
    return -sin_pi(-x);
  } else if (x < T1(0.5L)) {
    return std::sin(x * c10::numbers::pi_v<T2>);
  } else if (x < T1(1)) {
    return std::sin((T1(1) - x) * c10::numbers::pi_v<T2>);
  } else {
    if ((int(std::floor(x)) & 1) == 1) {
      if (x - std::floor(x) < T1(0.5L)) {
        return -1 * sin_pi(x - std::floor(x));
      } else {
        return -1 * sin_pi(T1(1) - (x - std::floor(x)));
      }
    } else {
      if (x - std::floor(x) < T1(0.5L)) {
        return +1 * sin_pi(x - std::floor(x));
      } else {
        return +1 * sin_pi(T1(1) - (x - std::floor(x)));
      }
    }
  }
}

template<typename T1>
std::complex<T1>
sin_pi(std::complex<T1> z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  return sin_pi(std::real(z)) * std::cosh(c10::numbers::pi_v<T3> * std::imag(z))
      + std::complex<T1>{0, 1} * cos_pi(std::real(z)) * std::sinh(c10::numbers::pi_v<T3> * std::imag(z));
}
}
}
}
}
