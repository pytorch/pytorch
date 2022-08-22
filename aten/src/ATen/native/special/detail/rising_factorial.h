#pragma once

#include <ATen/native/special/detail/factorial.h>
#include <ATen/native/special/detail/ln_factorial.h>
#include <ATen/native/special/detail/ln_gamma.h>
#include <ATen/native/special/detail/numeric_t.h>
#include <ATen/native/special/ln_gamma_sign.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
rising_factorial(T1 a, int n) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::isnan(a)) {
    return std::numeric_limits<T3>::quiet_NaN();
  } else if (n == 0) {
    return T1(1);
  } else if (std::abs(a - std::nearbyint(a)) < std::numeric_limits<T3>::epsilon()) {
    if (int(std::nearbyint(a)) < static_cast<int>(c10::numbers::factorials_size<T3>()) && a + n < static_cast<int>(c10::numbers::factorials_size<T3>())) {
      return factorial<T3>(int(std::nearbyint(a)) + n - T3(1)) / factorial<T3>(int(std::nearbyint(a)) - T3(1));
    } else {
      return std::exp(ln_factorial<T3>(int(std::nearbyint(a)) + n - T3(1)) - ln_factorial<T3>(int(std::nearbyint(a))) - T3(1));
    }
  } else if (std::abs(a) < c10::numbers::factorials_size<T3>() && std::abs(a + n) < c10::numbers::factorials_size<T3>()) {
    auto p = a;

    for (int j = 1; j < n; j++) {
      p = p * (a + j);
    }

    return p;
  } else {
    if (ln_gamma(a + n) - ln_gamma(a) < std::log(std::numeric_limits<T1>::max())) {
      return at::native::special::ln_gamma_sign(a + n) * at::native::special::ln_gamma_sign(a) * std::exp(ln_gamma(a + n) - ln_gamma(a));
    } else {
      return at::native::special::ln_gamma_sign(a + n) * at::native::special::ln_gamma_sign(a) * std::numeric_limits<T1>::infinity();
    }
  }
}

template<typename T1>
T1
rising_factorial(T1 a, T1 n) {
  if (std::isnan(n) || std::isnan(a)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (n == T1(0)) {
    return T1(1);
  } else if (std::abs(n - std::nearbyint(n)) < std::numeric_limits<T1>::epsilon()) {
    return rising_factorial(a, int(std::nearbyint(n)));
  } else if (ln_gamma(a + n) - ln_gamma(a) < std::log(std::numeric_limits<T1>::max())) {
    return at::native::special::ln_gamma_sign(a + n) * at::native::special::ln_gamma_sign(a) * std::exp(ln_gamma(a + n) - ln_gamma(a));
  } else {
    return at::native::special::ln_gamma_sign(a + n) * at::native::special::ln_gamma_sign(a) * std::numeric_limits<T1>::infinity();
  }
}
}
}
}
}
