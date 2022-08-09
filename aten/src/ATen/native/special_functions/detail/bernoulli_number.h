#pragma once

#include "bernoulli.h"

namespace at::native::special_functions::detail {
template<typename T1>
constexpr T1
bernoulli_number(unsigned int n) {
  return bernoulli_series<T1>(n);
}
}
