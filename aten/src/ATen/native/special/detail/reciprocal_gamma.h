#pragma once

#include <ATen/native/special/detail/complex.h>
#include <ATen/native/special/detail/gamma.h>
#include <ATen/native/special/detail/is_integer.h>
#include <ATen/native/special/detail/numeric_t.h>
#include <ATen/native/special/detail/promote_t.h>
#include <ATen/native/special/sin_pi.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
reciprocal_gamma(T1 z) {
  using T2 = numeric_t<T1>;

  if (std::isnan(z)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    const auto is_integer_z = is_integer(z);
    
    if (is_integer_z) {
      const auto integer_z = is_integer_z();

      if (integer_z <= 0) {
        return T1(0);
      } else if (integer_z < int(c10::numbers::factorials_size<T2>())) {
        return T1(1) / T2(c10::numbers::factorials_v[integer_z - 1]);
      } else {
        auto p = int(c10::numbers::factorials_size<T2>());
  
        auto q = T1(1) / T2(c10::numbers::factorials_v[integer_z - 1]);
  
        while (p < integer_z && std::abs(q) > T2(0)) {
          p++;

          q = q / T2(p);
        }
  
        return q;
      }
    } else if (std::real(z) > T2(1)) {
      auto p = int(std::floor(std::real(z)));

      auto q = z - T1(p);
  
      auto r = gamma_reciprocal_series(q);
  
      while (std::real(z) > T2(1) && std::abs(r) > 0) {
        z = z - T2(1);

        r = r / z;
      }
  
      return r;
    } else if (std::real(z) > T2(0)) {
      return gamma_reciprocal_series(z);
    } else {
      return at::native::special::sin_pi(z) * gamma(T1(1) - z) / c10::numbers::pi_v<T2>;
    }
  }
}
}
}
}
}
