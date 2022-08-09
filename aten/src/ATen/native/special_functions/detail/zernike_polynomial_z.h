#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/detail/radial_polynomial_r.h>

namespace at::native::special_functions::detail {
template<typename T1>
promote_t<T1>
zernike_polynomial_z(unsigned int n, int m, T1 rho, T1 phi) {
  if (std::isnan(rho) || std::isnan(phi)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (m >= 0) {
    return radial_polynomial_r(n, std::abs(m), rho) * std::cos(m * phi);
  } else {
    return radial_polynomial_r(n, std::abs(m), rho) * std::sin(m * phi);
  }
}
}

