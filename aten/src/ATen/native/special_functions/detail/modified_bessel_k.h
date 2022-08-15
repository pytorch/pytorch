#pragma once

#include "modified_bessel.h"

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
modified_bessel_k(T1 n, T1 x) {
  if (x < T1(0)) {
    throw std::domain_error("modified_bessel_k: Argument < 0");
  } else if (std::isnan(n) || std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    return modified_bessel(n, x).k;
  }
}
}
}
}
}
