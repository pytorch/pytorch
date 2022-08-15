#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename Tp>
Tp
log_binomial_coefficient_sign(Tp nu, unsigned int k) {
  auto n = std::nearbyint(nu);
  if (n >= 0 && nu == n) {
    return Tp{1};
  } else {
    return log_gamma_sign(Tp(1) + nu)
        * log_gamma_sign(Tp(1 + k))
        * log_gamma_sign(Tp(1 - k) + nu);
  }
}

template<typename Tp>
std::complex<Tp>
log_binomial_coefficient_sign(std::complex<Tp> n, unsigned int k) {
  return std::complex < Tp > {1};
}
}
}
}
}
