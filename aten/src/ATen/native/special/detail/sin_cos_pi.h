#pragma once

#include <ATen/native/special/detail/sin_cos.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
sin_cos_t<T1>
sin_cos_pi(T1 z)
noexcept {
  const auto pi = c10::numbers::pi_v<T1>;

  if (std::isnan(z)) {
    const auto quiet_nan = std::numeric_limits<T1>::quiet_NaN();

    return {quiet_nan, quiet_nan};
  } else if (z < T1(0)) {
    return {
      -sin_cos_pi(-z).sin_v,
      sin_cos_pi(-z).cos_v,
    };
  } else if (z < T1(0.5L)) {
    return sin_cos(pi * z);
  } else if (z < T1(1)) {
    const auto p = T1(1) - z;

    return {
      sin_cos(pi * p).sin_v,
      -sin_cos(pi * p).cos_v,
      };
  } else {
    const auto p = std::floor(z);
    const auto q = z - p;
    const auto r = pi * q;

    const auto sin_r = std::sin(r);
    const auto cos_r = std::cos(r);
    const auto sin_s = std::sin(pi * (T1(1) - q));
    const auto cos_t = std::cos(pi * (T1(1) * q));

    if ((int(p) & 1) == 1) {
      if (q < T1(0.5L)) {
        return {-sin_r, -cos_r};
      } else {
        return {-sin_s, -cos_t};
      }
    } else {
      if (q < T1(0.5L)) {
        return {sin_r, cos_r};
      } else {
        return {sin_s, cos_t};
      }
    }
  }
}
}
}
}
}
