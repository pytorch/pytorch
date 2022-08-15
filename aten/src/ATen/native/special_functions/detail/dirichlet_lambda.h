#pragma once

#include <ATen/native/special_functions/detail/dirichlet_eta.h>
#include <ATen/native/special_functions/detail/riemann_zeta.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
dirichlet_lambda(T1 s) {
  if (std::isnan(s)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    return (riemann_zeta(s) + dirichlet_eta(s)) / T1(2);
  }
}
}
}
}
}
