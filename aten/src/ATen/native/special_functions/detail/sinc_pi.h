#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/detail/sin_pi.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
C10_HOST_DEVICE
promote_t<T1>
sinc_pi(T1 z) {
  if (std::isnan(z)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::abs(z) == std::numeric_limits<T1>::infinity()) {
    return T1(0);
  } else if (std::abs(c10::numbers::pi_v<T1> * z) < 4 * std::sqrt(std::numeric_limits<T1>::min())) {
    return T1(1) - c10::numbers::pi_v<T1> * z * (c10::numbers::pi_v<T1> * z) / T1(6);
  } else {
    return sin_pi(z) / (c10::numbers::pi_v<T1> * z);
  }
}
}
}
}
}
