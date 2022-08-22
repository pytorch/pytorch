#pragma once

#include <ATen/native/special/detail/ln_double_factorial.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
constexpr T1
double_factorial(int n) {
  if (n < 0) {
    if (n % 2 == 1) {
      if (-n <= int(c10::numbers::negative_double_factorials_size<T1>())) {
        return c10::numbers::negative_double_factorials_v[-(1 + n) / 2].factorial;
      } else {
        return std::exp(ln_double_factorial(T1(n)));
      }
    } else {
      return std::numeric_limits<T1>::quiet_NaN();
    }
  } else if (n < int(c10::numbers::double_factorials_size<T1>())) {
    return c10::numbers::factorials_v[n];
  } else {
    return std::numeric_limits<T1>::infinity();
  }
}
}
}
}
}
