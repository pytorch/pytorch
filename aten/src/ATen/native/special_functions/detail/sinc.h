#pragma once

namespace at::native::special_functions::detail {
template<typename T1>
promote_t<T1>
sinc(T1 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::abs(x) == std::numeric_limits<T1>::infinity()) {
    return T1(0);
  } else if (std::abs(x) < std::sqrt(std::numeric_limits<T1>::min())) {
    return T1(1) - x * x / T1(6);
  } else {
    return std::sin(x) / x;
  }
}
}
