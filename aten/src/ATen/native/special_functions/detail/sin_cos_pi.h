#pragma once

#include <ATen/native/special_functions/detail/sin_cos.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
sin_cos_t<T1>
sin_cos_pi(T1 x)
noexcept {
  if (std::isnan(x)) {
    return {
        std::numeric_limits<T1>::quiet_NaN(),
        std::numeric_limits<T1>::quiet_NaN(),
    };
  } else if (x < T1(0)) {
    return {
        -sin_cos_pi(-x).sin_v,
        +sin_cos_pi(-x).cos_v,
    };
  } else if (x < T1(0.5L)) {
    return sin_cos(c10::numbers::pi_v<T1> * x);
  } else if (x < T1(1)) {
    return {
        +sin_cos(c10::numbers::pi_v<T1> * (T1(1) - x)).sin_v,
        -sin_cos(c10::numbers::pi_v<T1> * (T1(1) - x)).cos_v,
    };
  } else {
    if ((int(std::floor(x)) & 1) == 1) {
      if (x - std::floor(x) < T1(0.5L)) {
        return {
            T1(-1) * std::sin(c10::numbers::pi_v<T1> * (x - std::floor(x))),
            T1(-1) * std::cos(c10::numbers::pi_v<T1> * (x - std::floor(x))),
        };
      } else {
        return {
            T1(-1) * std::sin(c10::numbers::pi_v<T1> * (T1(1) - (x - std::floor(x)))),
            T1(-1) * std::cos(c10::numbers::pi_v<T1> * (T1(1) * (x - std::floor(x)))),
        };
      }
    } else {
      if (x - std::floor(x) < T1{0.5L}) {
        return {
            T1{+1} * std::sin(c10::numbers::pi_v<T1> * (x - std::floor(x))),
            T1{+1} * std::cos(c10::numbers::pi_v<T1> * (x - std::floor(x))),
        };
      } else {
        return {
            T1{+1} * std::sin(c10::numbers::pi_v<T1> * (T1(1) - (x - std::floor(x)))),
            T1{+1} * std::cos(c10::numbers::pi_v<T1> * (T1(1) * (x - std::floor(x)))),
        };
      }
    }
  }
}
}
}
}
}
