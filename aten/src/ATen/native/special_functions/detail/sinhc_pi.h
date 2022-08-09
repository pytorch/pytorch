#pragma once

#include <c10/util/numbers.h>

namespace at::native::special_functions {
template<typename T1>
inline constexpr detail::promote_t<T1>
sinh_pi(T1 x);

template<typename T1>
inline constexpr std::complex<T1>
sinh_pi(std::complex<T1> z);
}

namespace at::native::special_functions::detail {
template<typename T1>
promote_t<T1>
sinhc_pi(T1 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::abs(c10::numbers::pi_v<T1> * x) < T1(4) * std::sqrt(std::numeric_limits<T1>::min())) {
    return T1(1) + c10::numbers::pi_v<T1> * x * (c10::numbers::pi_v<T1> * x) / T1(6);
  } else {
    return at::native::special_functions::sinh_pi(x) / (c10::numbers::pi_v<T1> * x);
  }
}
}
