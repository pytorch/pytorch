#pragma once

#include "polylog.h"

namespace at::native::special_functions::detail {
template<typename T1>
T1
dirichlet_beta(std::complex<T1> z) {
  if (std::isnan(z)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (is_real(z)) {
    return std::imag(polylog(z.real(), std::complex<T1>{0, 1}));
  } else {
    throw std::domain_error("dirichlet_beta: Bad argument.");
  }
}

template<typename T1>
T1
dirichlet_beta(T1 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    return std::imag(polylog(x, std::complex<T1>{0, 1}));
  }
}
}
