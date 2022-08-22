#pragma once

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
lower_incomplete_gamma(T1 a, T1 x) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::isnan(a) || std::isnan(x) || is_integer(a) && (is_integer(a)() <= 0)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::real(x) < std::real(a + T3(1))) {
    return std::exp(gamma_series(a, x).second) * gamma_series(a, x).first;
  } else {
    return std::exp(gamma_continued_fraction(a, x).second) * (T1(1) - gamma_continued_fraction(a, x).first);
  }
}
}
}
}
}
