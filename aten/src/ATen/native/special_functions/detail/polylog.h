#pragma once

#include "complex.h"
#include "exp_polylog.h"
#include "factorial.h"
#include "is_imag.h"
#include "reciprocal_gamma.h"
#include "riemann_zeta.h"
#include "zeta.h"

namespace at::native::special_functions::detail {
template<typename T1>
T1
polylog(T1 s, T1 x) {
  if (std::isnan(s) || std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (is_zero(x)) {
    return T1(0);
  } else if (is_integer(s, T1(5)) && is_integer(s, T1(5))() == 1) {
    return -std::log(T1(1) - x);
  } else if (is_integer(s, T1(5)) && is_integer(s, T1(5))() == 0) {
    return x / (T1(1) - x);
  } else if (is_equal(x, T1(-1))) {
    return std::real(exp_polylog(s, T1(0)) * (std::pow(T1(2), T1(1) - s) - T1(1)));
  } else if (x < T1(0)) {
    return std::real(exp_polylog(s, T1(2) * std::log(-x)) * std::pow(T1(2), T1(1) - s) - exp_polylog(s, std::log(-x)));
  } else {
    return std::real(exp_polylog(s, std::log(x)));
  }
}

template<typename T1>
std::complex<T1>
polylog(T1 s, std::complex<T1> w) {
  if (std::isnan(s) || std::isnan(w)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (is_real(w)) {
    return polylog(s, std::real(w));
  } else {
    return exp_polylog(s, std::log(w));
  }
}
}
