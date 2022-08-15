#pragma once

#include "bernoulli.h"

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
constexpr T1
bernoulli_number(unsigned int n) {
  return bernoulli_series<T1>(n);
}
}
}
}
}
