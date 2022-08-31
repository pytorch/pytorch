#pragma once

#include <ATen/native/special_functions/detail/cos_pi.h>
#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/sin_pi.h>
#include <c10/util/MathConstants.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
C10_HOST_DEVICE
T1
cosh_pi(T1 z) {
  using T2 = numeric_t<T1>;

  if (std::isnan(z)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (z < T1(0)) {
    return cosh_pi(-z);
  } else {
    return std::cosh(c10::pi<T2> * z);
  }
}

template<typename T1>
C10_HOST_DEVICE
c10::complex<T1>
cosh_pi(c10::complex<T1> z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  return std::cosh(c10::pi<T3> * std::real(z)) * cos_pi(std::imag(z)) + c10::complex<T1>(0, 1) * std::sinh(c10::pi<T3> * std::real(z)) * sin_pi(std::imag(z));
}
}
}
}
}
