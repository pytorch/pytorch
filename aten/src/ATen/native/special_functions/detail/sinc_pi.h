#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>
#include <c10/util/numbers.h>
#include <ATen/native/special_functions/sin_pi.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
promote_t<T1>
sinc_pi(T1 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::abs(x) == std::numeric_limits<T1>::infinity()) {
    return T1(0);
  } else if (std::abs(c10::numbers::pi_v<T1> * x) < T1(4) * std::sqrt(std::numeric_limits<T1>::min())) {
    return T1(1) - c10::numbers::pi_v<T1> * x * (c10::numbers::pi_v<T1> * x) / T1(6);
  } else {
    return at::native::special_functions::sin_pi(x) / (c10::numbers::pi_v<T1> * x);
  }
}
}
}
}
}
