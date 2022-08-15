#pragma once

#include <ATen/native/special_functions/detail/bessel.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
bessel_y(T1 x, T1 n) {
  if (x < T1(0)) {
    throw std::domain_error("negative `x`");
  } else if (std::isnan(n) || std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    return bessel(x, n).y;
  }
}
}
}
}
}
