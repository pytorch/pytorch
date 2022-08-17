#pragma once

#include <ATen/native/special_functions/detail/ln_gamma.h>
#include <ATen/native/special_functions/detail/negative_double_factorials.h>
#include <ATen/native/special_functions/cos_pi.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
constexpr T1
ln_double_factorial(T1 n) {
  return (n / T1(2)) * std::log(T1(2))
      + (at::native::special_functions::cos_pi(n) - T1(1)) * std::log(c10::numbers::pi_v<T1> / T1(2)) / T1(4)
      + ln_gamma(
          T1(1) + n / T1(2));
}

template<typename T1>
constexpr T1
log_double_factorial(int n) {
  if (n < 0) {
    if (n % 2 == 1) {
      if (-n <= static_cast<int>(negative_double_factorials_size<T1>())) {
        return NEGATIVE_DOUBLE_FACTORIALS[-(1 + n) / 2].log_factorial;
      } else {
        return ln_double_factorial(T1(n));
      }
    } else {
      return std::numeric_limits<T1>::quiet_NaN();
    }
  } else if (n < static_cast<int>(c10::numbers::double_factorials_size<T1>())) {
    return c10::numbers::log_double_factorials_v[n];
  } else {
    return ln_double_factorial(T1(n));
  }
}
}
}
}
}
