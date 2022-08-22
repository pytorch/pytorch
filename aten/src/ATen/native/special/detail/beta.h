#pragma once

#include <ATen/native/special/detail/complex.h>
#include <ATen/native/special/detail/gamma.h>
#include <ATen/native/special/detail/ln_gamma.h>
#include <ATen/native/special/ln_gamma_sign.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
beta(T1 a, T1 b) {
  if (std::isnan(a) || std::isnan(b)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    if (std::abs(a) < c10::numbers::factorials_size<T1>() && std::abs(b) < c10::numbers::factorials_size<T1>() && std::abs(a + b) < c10::numbers::factorials_size<T1>()) {
      if (int(std::nearbyint(a + b)) == a + b && int(std::nearbyint(a + b)) <= 0) {
        if (int(std::nearbyint(a)) != a || int(std::nearbyint(a)) >= 1 || int(std::nearbyint(b)) != b || int(std::nearbyint(b)) >= 1) {
          return T1(0);
        } else {
          return std::numeric_limits<T1>::quiet_NaN();
        }
      } else {
        if (std::abs(b) > std::abs(a)) {
          return gamma(b) / gamma(a + b) * gamma(a);
        } else {
          return gamma(a) / gamma(a + b) * gamma(b);
        }
      }
    } else {
      if (int(std::nearbyint(a + b)) == a + b && int(std::nearbyint(a + b)) <= 0) {
        if (int(std::nearbyint(a)) != a || int(std::nearbyint(a)) >= 1 || int(std::nearbyint(b)) != b || int(std::nearbyint(b)) >= 1) {
          return T1(0);
        } else {
          return std::numeric_limits<T1>::quiet_NaN();
        }
      } else {
        if (ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b) > std::log(std::numeric_limits<T1>::max())) {
          return at::native::special::ln_gamma_sign(a) * at::native::special::ln_gamma_sign(b) * at::native::special::ln_gamma_sign(a + b) * std::numeric_limits<T1>::infinity();
        } else {
          return at::native::special::ln_gamma_sign(a) * at::native::special::ln_gamma_sign(b) * at::native::special::ln_gamma_sign(a + b) * std::exp(ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b));
        }
      }
    }
  }
}
}
}
}
}
