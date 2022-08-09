#pragma once

#include <ATen/native/special_functions/detail/expint.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
hyperbolic_cosine_integral_chi(const T1 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (x == T1(0)) {
    return T1(0);
  } else {
    return (expint_Ei(x) - expint_E1(x)) / T1(2);
  }
}
}
