#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
clausen_sl(unsigned int m, std::complex<T1> z) {
  if (std::isnan(z)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (m == 0) {
    throw std::domain_error("non-positive order `m`");
  } else {
    if (m & 1) {
      return std::imag(exp_polylog(T1(m), std::complex < T1 > {0, 1} * z));
    } else {
      return std::real(exp_polylog(T1(m), std::complex < T1 > {0, 1} * z));
    }
  }
}

template<typename T1>
T1
clausen_sl(unsigned int m, T1 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (m == 0) {
    throw std::domain_error("non-positive order `m`");
  } else {
    if (m & 1) {
      return std::imag(exp_polylog(T1(m), std::complex < T1 > {0, 1} * x));
    } else {
      return std::real(exp_polylog(T1(m), std::complex < T1 > {0, 1} * x));
    }
  }
}
}
}
}
}
