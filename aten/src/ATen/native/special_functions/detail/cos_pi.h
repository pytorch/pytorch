#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
cos_pi(T1 x) {
  using T2 = numeric_t<T1>;

  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (x < T1(0)) {
    return cos_pi(-x);
  } else if (x < T1{0.5L}) {
    return std::cos(x * c10::numbers::pi_v<T2>);
  } else if (x < T1(1)) {
    return -std::cos((T1(1) - x) * c10::numbers::pi_v<T2>);
  } else {
    return ((int(std::floor(x)) & 1) == 1 ? -1 : +1) * cos_pi(x - std::floor(x));
  }
}

template<typename T1>
std::complex<T1>
cos_pi(std::complex<T1> z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  const auto pi_v = c10::numbers::pi_v<T3>;

  const auto real_z = std::real(z);
  const auto imag_z = std::imag(z);

  return cos_pi(real_z) * std::cosh(pi_v * imag_z) - std::complex<T1>{0, 1} * sin_pi(real_z) * std::sinh(pi_v * imag_z);
}
}
}
}
}
