#pragma once

#include <ATen/native/special/detail/is_complex_v.h>
#include <ATen/native/special/detail/spouge_approximation.h>
#include <ATen/native/special/sin_pi.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
inline constexpr
T1
spouge_ln_gamma_approximation(T1 z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  auto a = T3(spouge_approximation<T3>::size + 1);

  constexpr auto is_not_complex = !is_complex_v<T2>;

  if (std::real(z) < T3(-1)) {
    if (is_not_complex) {
      return c10::numbers::lnpi_v<T3> - std::log(std::abs(at::native::special::sin_pi(z))) - spouge_ln_gamma_approximation(-T3(1) - z);
    } else {
      return c10::numbers::lnpi_v<T3> - std::log(at::native::special::sin_pi(z)) - spouge_ln_gamma_approximation(-T3(1) - z);
    }
  } else {
    T2 p = c10::numbers::sqrttau_v<T3>;

    for (auto j = 0; j < spouge_approximation<T3>::size; j++) {
      p = p + (spouge_approximation<T3>::coefficients[j] / (z + T3(j + 1)));
    }

    if (is_not_complex) {
      p = std::abs(p);
    }

    return std::log(p) + (z + T3(0.5L)) * std::log(z + a) - (z + a);
  }
}
}
}
}
}
