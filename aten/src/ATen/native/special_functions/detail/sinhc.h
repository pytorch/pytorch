#pragma once

namespace at::native::special_functions::detail {
template<typename T1>
promote_t<T1>
sinhc(T1 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::abs(x) < T1(4) * std::sqrt(std::numeric_limits<T1>::min())) {
    return T1(1) + x * x / T1(6);
  } else {
    return std::sinh(x) / x;
  }
}
}
