#pragma once

#include <c10/util/numbers.h>
#include <ATen/native/special_functions/detail/double_factorials.h>
#include <ATen/native/special_functions/detail/ln_gamma.h>

namespace at::native::special_functions::detail {
template<typename T1>
constexpr T1
ln_factorial(unsigned int n) {
  if (n < c10::numbers::factorials_size<T1>) {
    return c10::numbers::log_factorials_v[n];
  } else if (n < DOUBLE_FACTORIALS_SIZE < T1 >) {
    return DOUBLE_FACTORIALS[n].log_factorial + DOUBLE_FACTORIALS[n - 1].log_factorial;
  } else {
    return ln_gamma(T1(n + 1));
  }
}
}
