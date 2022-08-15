#pragma once

#include "laguerre_polynomial_l.h"

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1, typename T2>
T2
associated_laguerre_polynomial_l(unsigned int n, T1 alpha, T2 x) {
  return laguerre_polynomial_l<T1, T2>(n, alpha, x);
}
}
}
}
}
