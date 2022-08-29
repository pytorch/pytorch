#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
C10_HOST_DEVICE
promote_t<T1>
sinhc(T1 z) {
  if (std::isnan(z)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::abs(z) < 4 * std::sqrt(std::numeric_limits<T1>::min())) {
    return T1(1) + z * z / T1(6);
  } else {
    return std::sinh(z) / z;
  }
}
}
}
}
}
