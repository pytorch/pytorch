#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
inline constexpr
promote_t<T1>
sinc_pi(T1 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::abs(x) == std::numeric_limits<T1>::infinity()) {
    return T1{0};
  } else {
    if (std::abs(c10::pi<T1> * x) < T1{4} * std::sqrt(std::numeric_limits<T1>::min())) {
      return T1{1} - c10::pi<T1> * x * (c10::pi<T1> * x) / T1{6};
    } else {
      return mmath::sin_pi(x) / (c10::pi<T1> * x);
    }
  }
} // promote_t<T1> sinc_pi(T1 x)
} // namespace detail
} // namespace special_functions
} // namespace native
} // namespace at
