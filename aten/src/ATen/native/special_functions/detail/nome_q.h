#pragma once

#include <c10/util/numbers.h>

#include <ATen/native/special_functions/complete_elliptic_integral_k.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
nome_q(T1 k) {
  if (std::isnan(k)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::abs(k) > T1(1)) {
    throw std::domain_error("nome_q: argument k out of range");
  } else if (k < std::pow(T1(67) * std::numeric_limits<T1>::epsilon(), T1{0.125L})) {
    return k * k * ((T1(1) / T1(16)) + k * k * ((T1(1) / T1(32))
        + k * k * ((T1(21) / T1(1024)) + k * k * ((T1(31) / T1(2048)) + k * k * (T1(6257) / T1(524288))))));
  } else {
    return std::exp(-c10::numbers::pi_v<T1>
                        * at::native::special_functions::complete_elliptic_integral_k(std::sqrt(T1(1) - k * k))
                        / at::native::special_functions::complete_elliptic_integral_k(k));
  }
}
}
