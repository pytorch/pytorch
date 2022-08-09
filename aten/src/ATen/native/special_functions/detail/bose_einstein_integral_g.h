#pragma once

namespace at::native::special_functions::detail {
template<typename T1, typename T2>
T2
bose_einstein_integral_g(T1 s, T2 x) {
  if (std::isnan(s) || std::isnan(x)) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else if (s <= T1(0) && x < T2(0)) {
    throw std::domain_error("bose_einstein_integral_g: Order must be greater than 0");
  } else {
    return std::real(polylog_exp(s + T1(1), x));
  }
}
}
