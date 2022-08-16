#pragma once

#include "ln_double_factorial.h"

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
constexpr T1
double_factorial(int n) {
  if (n < 0) {
    if (n % 2 == 1) {
      if (-n <= int(NEGATIVE_DOUBLE_FACTORIALS_SIZE < T1 > )) {
        return NEGATIVE_DOUBLE_FACTORIALS[-(1 + n) / 2].factorial;
      } else {
        return std::exp(ln_double_factorial(T1(n)));
      }
    } else {
      return std::numeric_limits<T1>::quiet_NaN();
    }
  } else if (n < int(c10::numbers::DOUBLE_FACTORIALS_SIZE < T1 > )) {
    return c10::numbers::double_factorials_v[n];
  } else {
    return std::numeric_limits<T1>::infinity();
  }
}
}
}
}
}
