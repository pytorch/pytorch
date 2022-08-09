#pragma once

#include <cmath>
#include <complex>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
log_gamma_sign(T1 x) {
  if (x >= T1(0)) {
    return T1(1);
  } else if (x == std::nearbyint(x)) {
    return T1(0);
  } else if (int(-x) % 2 == 0) {
    return -T1(1);
  } else {
    return +T1(1);
  }
}

template<typename T1>
std::complex<T1>
log_gamma_sign(std::complex<T1> z) {
  return std::complex<T1>{1};
}
}
}
}
}
