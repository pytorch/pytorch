#pragma once

#include <ATen/native/special/detail/numeric_t.h>
#include <ATen/native/special/detail/tan_pi.h>
#include <c10/util/complex.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
tanh_pi(T1 x) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  return std::tanh(c10::numbers::pi_v<T3> * x);
}

template<typename T1>
c10::complex<T1>
tanh_pi(c10::complex<T1> z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  const auto real_z = std::real(z);
  const auto imag_z = std::imag(z);

  const auto pi_real_z = c10::numbers::pi_v<T3> * real_z;

  const auto tanh_pi_real_z = std::tanh(pi_real_z);

  const auto tan_pi_imag_z = tan_pi(imag_z);

  const auto tanh_pi_real_z_tan_pi_imag_z = tanh_pi_real_z * tan_pi_imag_z;

  return (tanh_pi_real_z + c10::complex<T1>(0, 1) * tan_pi_imag_z) / (T2(1) + c10::complex<T1>(0, 1) * tanh_pi_real_z_tan_pi_imag_z);
}
}
}
}
}
