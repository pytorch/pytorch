#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
logarithmic_integral_li(const T1 x) {
  if (std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (std::abs(x) == T1(1)) {
    return std::numeric_limits<T1>::infinity();
  } else {
    return expint(std::log(x));
  }
}
}
}
}
}
