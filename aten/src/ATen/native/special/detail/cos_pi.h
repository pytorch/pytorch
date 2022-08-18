#pragma once

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
cos_pi(T1 x) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (x < T1(0)) {
    return cos_pi(-x);
  } else if (x < T1(0.5L)) {
    return std::cos(x * c10::numbers::pi_v<T2>);
  } else if (x < T1(1)) {
    return -std::cos((T1(1) - x) * c10::numbers::pi_v<T2>);
  } else if ((int(std::floor(x)) & 1) == 1) {
    return -1 * cos_pi(x - std::floor(x));
  } else {
    return +1 * cos_pi(x - std::floor(x));
  }
}

template<typename T1>
c10::complex<T1>
cos_pi(c10::complex<T1> z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  const auto real_z = std::real(z);

  const auto cos_pi_real_z = cos_pi(real_z);
  const auto sin_pi_real_z = sin_pi(real_z);

  const auto pi_imag_z = c10::numbers::pi_v<T3> * std::imag(z);

  const auto cosh_pi_imag_z = std::cosh(pi_imag_z);
  const auto sinh_pi_imag_z = std::sinh(pi_imag_z);

  const auto cos_pi_real_z_cosh_pi_imag_z = cos_pi_real_z * cosh_pi_imag_z;
  const auto sin_pi_real_z_sinh_pi_imag_z = sin_pi_real_z * sinh_pi_imag_z;

  return cos_pi_real_z_cosh_pi_imag_z - c10::complex<T1>(0, 1) * sin_pi_real_z_sinh_pi_imag_z;
}
}
}
}
}
