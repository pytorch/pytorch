#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/sin_pi.h>
#include <c10/util/numbers.h>
#include <ATen/native/special_functions/detail/is_integer.h>
#include <ATen/native/special_functions/detail/gamma.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
reciprocal_gamma(T1 a) {
  using T2 = numeric_t<T1>;

  if (std::isnan(a)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (is_integer(a)) {
    if (is_integer(a)() <= 0) {
      return T1(0);
    } else if (is_integer(a)() < int(c10::numbers::factorials_size<T2>())) {
      return T1(1) / T2(c10::numbers::factorials_v[is_integer(a)() - 1]);
    } else {
      auto k = int(c10::numbers::factorials_size<T2>());

      auto g = T1(1) / T2(c10::numbers::factorials_v[is_integer(a)() - 1]);

      while (k < is_integer(a)() && std::abs(g) > T2(0)) {
        g /= T2(k++);
      }

      return g;
    }
  } else if (std::real(a) > T2(1)) {
    auto n = int(std::floor(std::real(a)));
    auto nu = a - T1(n);

    auto g = gamma_reciprocal_series(nu);

    while (std::real(a) > T2(1) && std::abs(g) > T1(0)) {
      g /= (a -= T2(1));
    }

    return g;
  } else if (std::real(a) > T2(0)) {
    return gamma_reciprocal_series(a);
  } else {
    return at::native::special_functions::sin_pi(a) * gamma(T1(1) - a) / c10::numbers::pi_v<T2>;
  }
}
}
}
}
}
