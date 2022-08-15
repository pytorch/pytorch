#pragma once

#include <complex>

#include <ATen/native/special_functions/detail/polylog.h>
#include <ATen/native/special_functions/detail/is_integer.h>
#include <ATen/native/special_functions/detail/sin_pi.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
std::complex<T1>
dirichlet_eta(std::complex<T1> s) {
  if (std::isnan(s)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (is_real(s)) {
    return -polylog(std::real(s), T1(-1));
  } else {
    throw std::domain_error("dirichlet_eta: Bad argument");
  }
}

template<typename T1>
T1
dirichlet_eta(T1 s) {
  if (std::isnan(s)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (s < T1(0)) {
    if (is_integer(s, T1(5)) && (is_integer(s, T1(5))() % 2 == 0)) {
      return T1(0);
    } else {
      return T1(2) * (T1(1) - std::pow(T1(2), -(T1(1) - s))) / (T1(1) - T1(2) * std::pow(T1(2), -(T1(1) - s)))
          * std::pow(c10::numbers::pi_v<T1>, -(T1(1) - s)) * s * sin_pi(s / T1(2)) * gamma(-s)
          * dirichlet_eta(T1(1) - s);
    }
  } else {
    return -std::real(polylog(s, T1(-1)));
  }
}
}
}
}
}
