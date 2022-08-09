#pragma once

#include <ATen/native/special_functions/detail/ln_factorial.h>
#include <ATen/native/special_functions/detail/numeric_t.h>
#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
ln_binomial_coefficient(unsigned int n, unsigned int k) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (k > n) {
    return -T2(std::numeric_limits<T3>::infinity());
  } else if (k == 0 || k == n) {
    return T2(0);
  } else if (n < c10::numbers::factorials_size<T3> && k < c10::numbers::factorials_size<T3>) {
    return ln_factorial<T2>(n) - ln_factorial<T2>(k) - ln_factorial<T2>(n - k);
  } else {
    return ln_gamma(T2(1 + n)) - ln_gamma(T2(1 + k)) - ln_gamma(T2(1 + n - k));
  }
}

template<typename T1>
T1
log_binomial_coefficient(T1 n, unsigned int k) {
  if (std::nearbyint(n) >= 0 && n == std::nearbyint(n)) {
    return ln_binomial_coefficient<T1>(static_cast<unsigned int>(std::nearbyint(n)), k);
  } else {
    return ln_gamma(T1(1) + n) - ln_gamma(T1(1 + k)) - ln_gamma(T1(1 - k) + n);
  }
}
}
