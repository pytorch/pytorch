#pragma once

#include <ATen/native/special_functions/detail/numeric.h>
#include <c10/util/MathConstants.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
C10_HOST_DEVICE
T1
tan_pi(T1 z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  return std::tan(c10::pi<T3> * (z - std::floor(z)));
}

template<typename T1>
C10_HOST_DEVICE
c10::complex<T1>
tan_pi(c10::complex<T1> z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  return (tan_pi(std::real(z)) + c10::complex<T1>(0, 1) * std::tanh(c10::pi<T3> * std::imag(z))) / (T2(1) - c10::complex<T1>(0, 1) * tan_pi(std::real(z)) * std::tanh(c10::pi<T3> * std::imag(z)));
}
}
}
}
}
