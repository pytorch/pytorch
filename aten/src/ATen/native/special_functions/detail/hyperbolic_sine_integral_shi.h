#pragma once

#include <ATen/native/special_functions/detail/expint.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
hyperbolic_sine_integral_shi(const T1 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    return (expint_Ei(x) + expint_E1(x)) / T1(2);
  }
}
}
}
}
}
