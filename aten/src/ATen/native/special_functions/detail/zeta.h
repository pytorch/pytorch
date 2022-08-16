#pragma once

#include <ATen/native/special_functions/detail/is_complex_v.h>
#include <ATen/native/special_functions/detail/numeric_t.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
exp2(T1 x) {
  const auto p = is_complex_v<T1>;

  if (p) {
    return std::pow(numeric_t<T1>(2), x);
  } else {
    return std::exp2(x);
  }
}
}
}
}
}
