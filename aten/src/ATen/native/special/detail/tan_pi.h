#pragma once

#include <ATen/native/special/detail/numeric_t.h>
#include <c10/util/complex.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
tan_pi(T1 x) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  return std::tan(c10::numbers::pi_v<T3> * (x - std::floor(x)));
}

template<typename T1>
c10::complex<T1>
tan_pi(c10::complex<T1> z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  const auto tan_pi_real_z = tan_pi(std::real(z));

  const auto pi_imag_z = c10::numbers::pi_v<T3> * std::imag(z);

  const auto tanh_pi_imag_z = std::tanh(pi_imag_z);

  const auto tan_pi_real_z_tanh_pi_imag_z = tan_pi_real_z * tanh_pi_imag_z;

  return (tan_pi_real_z + c10::complex<T1>(0, 1) * tanh_pi_imag_z) / (T2(1) - c10::complex<T1>(0, 1) * tan_pi_real_z_tanh_pi_imag_z);
}
}
}
}
}
