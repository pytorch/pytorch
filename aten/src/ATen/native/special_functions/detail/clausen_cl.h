#pragma once

#include "polylog.h"

namespace at::native::special_functions::detail {
template<typename T1>
T1
clausen_cl(unsigned int m, std::complex<T1> z) {
  if (std::isnan(z))
    return std::numeric_limits<T1>::quiet_NaN();
  else if (m == 0)
    throw std::domain_error("non-positive order `m`");
  else {
    const auto s_i = std::complex<T1>{0, 1};
    const auto ple = exp_polylog(T1(m), s_i * z);
    if (m & 1)
      return std::real(ple);
    else
      return std::imag(ple);
  }
}

template<typename T1>
T1
clausen_cl(unsigned int m, T1 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (m == 0)
    throw std::domain_error("non-positive order `m`");
  else {
    if (m & 1)
      return std::real(exp_polylog(T1(m), std::complex<T1>{0, 1} * x));
    else
      return std::imag(exp_polylog(T1(m), std::complex<T1>{0, 1} * x));
  }
}
}
