#pragma once

#include <ATen/native/special_functions/detail/falling_factorial.h>
#include <ATen/native/special_functions/detail/ln_factorial.h>
#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/is_integer.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
ln_falling_factorial(T1 a, T1 n) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::isnan(n) || std::isnan(a)) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else if (n == T1(0)) {
    return T1(0);
  } else if (is_integer(n)) {
    if (is_integer(a)) {
      if (is_integer(a)() < is_integer(n)()) {
        return -std::numeric_limits<T2>::infinity();
      } else {
        return ln_factorial<T2>(unsigned(is_integer(a)()))
            - ln_factorial<T2>(unsigned(is_integer(a)() - is_integer(n)()));
      }
    } else {
      return std::log(std::abs(falling_factorial(a, is_integer(n)())));
    }
  } else if (std::abs(a) < c10::numbers::factorials_size<T3> && std::abs(a - n) < c10::numbers::factorials_size<T3>) {
    return std::log(std::abs(falling_factorial(a, n)));
  } else {
    return ln_gamma(a + T1(1)) - ln_gamma(a - n + T1(1));
  }
}
}
}
}
}
