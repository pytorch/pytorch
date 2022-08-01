#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
inline constexpr
sin_cos_t<T1>
sin_cos_pi(T1 x) {
  if (std::isnan(x)) {
    return sin_cos_t<T1>{
        std::numeric_limits<T1>::quiet_NaN(),
        std::numeric_limits<T1>::quiet_NaN(),
    };
  } else if (x < T1{0}) {
    return sin_cos_t<T1>{
        -sin_cos_pi(-x).sin_v,
        +sin_cos_pi(-x).cos_v,
    };
  } else if (x < T1{0.5L}) {
    return sin_cos(c10::pi<T1> * x);
  } else if (x < T1{1}) {
    return sin_cos_t<T1>{
        +sin_cos(c10::pi<T1> * (T1{1} - x)).sin_v,
        -sin_cos(c10::pi<T1> * (T1{1} - x)).cos_v,
    };
  } else {
    if ((int(std::floor(x)) & 1) == 1) {
      if (x - std::floor(x) < T1{0.5L}) {
        return sin_cos_t<T1>{
            T1{-1} * std::sin(c10::pi<T1> * (x - std::floor(x))),
            T1{-1} * std::cos(c10::pi<T1> * (x - std::floor(x))),
        };
      } else {
        return sin_cos_t<T1>{
            T1{-1} * std::sin(c10::pi<T1> * (T1{1} - (x - std::floor(x)))),
            T1{-1} * std::cos(c10::pi<T1> * (T1{1} * (x - std::floor(x)))),
        };
      }
    } else {
      if (x - std::floor(x) < T1{0.5L}) {
        return sin_cos_t<T1>{
            T1{+1} * std::sin(c10::pi<T1> * (x - std::floor(x))),
            T1{+1} * std::cos(c10::pi<T1> * (x - std::floor(x))),
        };
      } else {
        return sin_cos_t<T1>{
            T1{+1} * std::sin(c10::pi<T1> * (T1{1} - (x - std::floor(x)))),
            T1{+1} * std::cos(c10::pi<T1> * (T1{1} * (x - std::floor(x)))),
        };
      }
    }
  }
} // sin_cos_t<T1> sin_cos_pi(T1 x)
} // namespace detail
} // namespace special_functions
} // namespace native
} // namespace at
