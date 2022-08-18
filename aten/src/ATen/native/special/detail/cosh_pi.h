#pragma once

#include <ATen/native/special/detail/cos_pi.h>
#include <ATen/native/special/detail/sin_pi.h>
#include <ATen/native/special/detail/numeric_t.h>
#include <c10/util/complex.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
cosh_pi(T1 x) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (x < T1(0)) {
    return cosh_pi(-x);
  } else {
    return std::cosh(c10::numbers::pi_v<T2> * x);
  }
}

template<typename T1>
c10::complex<T1>
cosh_pi(c10::complex<T1> z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  const auto imag_z = std::imag(z);

  const auto cos_pi_imag_z = cos_pi(imag_z);
  const auto sin_pi_imag_z = sin_pi(imag_z);

  const auto pi_real_z = c10::numbers::pi_v<T3> * std::real(z);

  const auto cosh_pi_real_z = std::cosh(pi_real_z);
  const auto sinh_pi_real_z = std::sinh(pi_real_z);

  const auto cos_pi_imag_z_cosh_pi_real_z = cosh_pi_real_z * cos_pi_imag_z;
  const auto sin_pi_imag_z_sinh_pi_real_z = sin_pi_imag_z * sinh_pi_real_z;

  return cos_pi_imag_z_cosh_pi_real_z + c10::complex<T1>(0, 1) * sin_pi_imag_z_sinh_pi_real_z;
}
}
}
}
}
