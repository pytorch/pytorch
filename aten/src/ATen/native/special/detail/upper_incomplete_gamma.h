#pragma once

#include <ATen/native/special/detail/complex.h>
#include <ATen/native/special/detail/gamma.h>
#include <ATen/native/special/detail/gamma_continued_fraction.h>
#include <ATen/native/special/detail/gamma_series.h>
#include <ATen/native/special/detail/is_integer.h>
#include <ATen/native/special/detail/numeric_t.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
upper_incomplete_gamma(T1 a, T1 z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::isnan(a) || std::isnan(z) || (is_integer(a) && is_integer(a)() <= 0)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::real(z) < std::real(a + T3(1))) {
    return std::exp(gamma_series(a, z).second) * (T1(1) - gamma_series(a, z).first);
  } else {
    return std::exp(gamma_continued_fraction(a, z).second) * gamma_continued_fraction(a, z).first;
  }
}
}
}
}
}
