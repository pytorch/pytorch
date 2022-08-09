#pragma once

#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename T1, typename T2>
T2
fermi_dirac_integral_f(T1 s, T2 x) {
  if (std::isnan(s) || std::isnan(x)) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else if (s <= T1(-1)) {
    throw std::domain_error("fermi_dirac_integral_f: Order must be greater than -1");
  } else {
    return -std::real(exp_polylog(s + T1(1), x + std::complex<T2>{0, 1} * c10::numbers::pi_v<T2>));
  }
}
}
