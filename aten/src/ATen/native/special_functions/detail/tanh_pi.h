#pragma once

#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
tanh_pi(T1 x) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  return std::tanh(c10::numbers::pi_v<T3> * x);
}

template<typename T1>
std::complex<T1>
tanh_pi(std::complex<T1> z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  return (std::tanh(c10::numbers::pi_v<T3> * std::real(z)) + std::complex<T1>{0, 1} * tan_pi(std::imag(z)))
      / (T2(1) + std::complex<T1>{0, 1} * std::tanh(c10::numbers::pi_v<T3> * std::real(z)) * tan_pi(std::imag(z)));
}
}
