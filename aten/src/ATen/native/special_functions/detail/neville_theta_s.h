#pragma once

#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
neville_theta_s(T1 k, T1 x) {
  using T2 = numeric_t<T1>;

  if (std::isnan(k) || std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::abs(k) > T1(1)) {
    throw std::domain_error("`k` out of range");
  } else {
    return std::sqrt(c10::numbers::pi_v<T2> / T2(2) / (k * std::sqrt(T1(1) - k * k)
        * at::native::special_functions::complete_elliptic_integral_k(k))) * theta_1(nome_q(k),
                                                                                       c10::numbers::pi_v<T2> / T2(2)
                                                                                           * x
                                                                                           / at::native::special_functions::complete_elliptic_integral_k(
                                                                                               k));
  }
}
}
}
}
}
