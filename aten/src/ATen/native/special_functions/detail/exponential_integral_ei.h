#pragma once

#include <cmath>

#include <ATen/native/special_functions/detail/expint.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
exponential_integral_ei(T1 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    return expint_Ei(x);
  }
}
}
}
}
}
