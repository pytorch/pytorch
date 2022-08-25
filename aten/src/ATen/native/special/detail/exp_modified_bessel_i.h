#pragma once

#include <ATen/native/special/detail/modified_bessel.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
exp_modified_bessel_i(T1 v, T1 z) {
  if (std::isnan(v) || std::isnan(z) || z < T1(0)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    return modified_bessel(v, z, true).i;
  }
}
}
}
}
}
