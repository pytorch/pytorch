#pragma once

#include <ATen/native/special_functions/detail/numeric.h>
#include <ATen/native/special_functions/detail/sin_pi.h>
#include <c10/util/complex.h>
#include <c10/util/MathConstants.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
C10_HOST_DEVICE
T1
sin_pi(T1 z);

template<typename T1>
C10_HOST_DEVICE
c10::complex<T1>
sin_pi(c10::complex<T1> z);

template<typename T1>
C10_HOST_DEVICE
T1
cos_pi(T1 z) {
  using T2 = numeric_t<T1>;

  if (std::isnan(z)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (z < T1(0)) {
    return cos_pi(-z);
  } else if (z < T1(0.5L)) {
    return std::cos(z * c10::pi<T2>);
  } else if (z < T1(1)) {
    return -std::cos((T1(1) - z) * c10::pi<T2>);
  } else {
    if ((int(std::floor(z)) & 1) == 1) {
      return -1 * cos_pi(z - std::floor(z));
    } else {
      return +1 * cos_pi(z - std::floor(z));
    }
  }
}

template<typename T1>
C10_HOST_DEVICE
c10::complex<T1>
cos_pi(c10::complex<T1> z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  return cos_pi(std::real(z)) * std::cosh(c10::pi<T3> * std::imag(z)) - c10::complex<T1>(0, 1) * sin_pi(std::real(z)) * std::sinh(c10::pi<T3> * std::imag(z));
}
}
}
}
}
