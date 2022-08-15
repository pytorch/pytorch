#pragma once

#include <cmath>

#include <ATen/native/special_functions/detail/jacobi_polynomial_p.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
radial_polynomial_r(unsigned int n, unsigned int m, T1 rho) {
  if (std::isnan(rho)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (int(n) - int(m) < 0 || (int(n) - int(m)) % 2 == 1) {
    return T1(0);
  } else if ((int(n) - int(m)) / 2 % 2 == 0) {
    return +1 * std::pow(rho, m)
        * jacobi_polynomial_p((int(n) - int(m)) / 2, T1(m), T1(0), T1(1) - T1(2) * rho * rho).P_n;
  } else {
    return -1 * std::pow(rho, m)
        * jacobi_polynomial_p((int(n) - int(m)) / 2, T1(m), T1(0), T1(1) - T1(2) * rho * rho).P_n;
  }
}
}
}
}
}
