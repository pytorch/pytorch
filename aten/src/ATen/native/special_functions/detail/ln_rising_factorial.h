#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/rising_factorial.h>
#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
log_rising_factorial(T1 a, T1 n) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::isnan(n) || std::isnan(a)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (n == T1(0)) {
    return T1(0);
  } else if (std::abs(a) < c10::numbers::factorials_size<T3> && std::abs(a + n) < c10::numbers::factorials_size<T3>) {
    return std::log(rising_factorial(a, n));
  } else {
    return ln_gamma(a + n) - ln_gamma(a);
  }
}
}
