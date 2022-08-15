#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/is_integer.h>
#include <ATen/native/special_functions/detail/gamma.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
upper_incomplete_gamma(T1 a, T1 x) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::isnan(a) || std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (is_integer(a) && is_integer(a)() <= 0) {
    throw std::domain_error("non-positive integer `a`");
  } else if (std::real(x) < std::real(a + T3(1))) {
    const auto g = gamma_series(a, x);

    return std::exp(g.second) * (T1(1) - g.first);
  } else {
    const auto g = gamma_continued_fraction(a, x);

    return std::exp(g.second) * g.first;
  }
}
}
}
}
}
